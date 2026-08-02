//! # BM3D-DEB: Deblurring / Deconvolution Extension
//!
//! Rust implementation of the BM3D deblurring scheme of Dabov et al. (2008),
//! *"Image restoration by sparse 3D transform-domain collaborative filtering"*.
//!
//! This module is a pure **extension**: it adds new entry points and does not change
//! the behaviour or the signatures of the existing (non-deblurring) API. Code that
//! only uses [`crate::orchestration::bm3d_ring_artifact_removal`],
//! [`crate::pipeline::run_bm3d_step`] etc. is unaffected.
//!
//! ## Observation model
//!
//! ```text
//! z = y (*) v + n,    n ~ N(0, sigma^2) white
//! ```
//!
//! where `v` is a known point spread function (PSF) that sums to 1 and `(*)` is
//! convolution with the PSF centred at `(psf_rows / 2, psf_cols / 2)`.
//!
//! ## Algorithm
//!
//! **Stage 1 - Regularized Inverse (RI)**
//!
//! ```text
//! RI(f)  = conj(V(f)) / (|V(f)|^2 + lambda_RI)
//! z_RI   = IFFT( FFT(z) * RI )
//! S_RI(f)= sigma^2 * |RI(f)|^2          (noise PSD of z_RI, colored)
//! y_RI   = BM3D-hard-threshold(z_RI, S_RI)
//! ```
//!
//! **Stage 2 - Regularized Wiener Inverse (RWI)**
//!
//! ```text
//! S_y(f)  = |FFT(y_RI)|^2 / (rows * cols)      (pilot signal PSD)
//! RWI(f)  = conj(V) * S_y / (|V|^2 * S_y + alpha_RWI * sigma^2)
//! z_RWI   = IFFT( FFT(z) * RWI )
//! S_RWI(f)= sigma^2 * |RWI(f)|^2
//! y_hat   = BM3D-wiener(z_RWI, pilot = y_RI, S_RWI)
//! ```
//!
//! Both BM3D passes see *colored* noise. The frequency-domain noise PSD is converted
//! into a per-2D-transform-coefficient standard deviation (see [`patch_noise_sigma`])
//! and handed to the kernel through
//! [`crate::pipeline::run_bm3d_step_colored_noise`].
//!
//! ## Regularization parameters
//!
//! Both regularization parameters are dimensionless and **scale invariant**, so they do
//! not have to be retuned when the data range changes (unlike the raw constants of the
//! original MATLAB reference, which assume `[0, 255]` images):
//!
//! - `reg_ri` (`lambda_RI = reg_ri * sigma^2 / signal_power`): stage 1 uses a flat signal
//!   prior, so `signal_power` is the estimated variance of the (blurred) signal.
//!   The default `1.0` reproduces the classic `4e-4 * sigma^2` for a `[0, 255]` image
//!   with a typical variance of about `2500`.
//! - `reg_rwi` (`alpha_RWI`): multiplies the noise power in the Wiener denominator.
//!   `1.0` would be the statistically optimal Wiener inverse; the default `5e-3`
//!   deliberately under-regularizes (less bias, more noise) because the following BM3D
//!   pass removes the residual noise. This matches the reference implementation.
//!
//! ## Example
//!
//! ```no_run
//! use bm3d_core::deblur::{Bm3dDeblurConfig, bm3d_deblur, gaussian_psf};
//! use ndarray::Array2;
//!
//! let observed: Array2<f32> = Array2::zeros((256, 256)); // blurred + noisy input
//! let psf = gaussian_psf::<f32>(1.5, 1.5);
//! let mut config = Bm3dDeblurConfig::<f32>::default();
//! config.sigma = 0.01; // leave at 0.0 to auto-estimate
//! let restored = bm3d_deblur(observed.view(), psf.view(), &config).unwrap();
//! ```

use ndarray::{s, Array1, Array2, Array3, ArrayView2, ArrayView3, Axis};
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

use crate::float_trait::Bm3dFloat;
use crate::pipeline::{run_bm3d_step_colored_noise, Bm3dKernelConfig, Bm3dMode, Bm3dPlans};
use crate::transforms;
use crate::utils::estimate_robust_sigma;

// =============================================================================
// Constants
// =============================================================================

/// Default stage-1 regularization factor (see module docs).
const DEFAULT_REG_RI: f64 = 1.0;

/// Default stage-2 (Wiener) regularization factor.
const DEFAULT_REG_RWI: f64 = 5e-3;

/// Default hard thresholding coefficient (lambda_3D).
const DEFAULT_THRESHOLD: f64 = 2.7;

/// Default block matching patch size.
const DEFAULT_PATCH_SIZE: usize = 8;

/// Default stride between reference patches.
const DEFAULT_STEP_SIZE: usize = 4;

/// Default block matching search window.
const DEFAULT_SEARCH_WINDOW: usize = 24;

/// Default maximum number of matched patches per group.
const DEFAULT_MAX_MATCHES: usize = 16;

/// Lower bound of the per-coefficient noise variance, relative to its mean.
/// Prevents zero thresholds / degenerate Wiener weights at frequencies where the
/// inverted noise PSD vanishes.
const PSD_FLOOR_RATIO: f64 = 1e-6;

/// Absolute lower bound for the regularization term, relative to `|V(f)|^2 == 1` (DC).
/// Only relevant for (near) noise-free inputs, where the inverse filter would explode.
const MIN_REGULARIZATION: f64 = 1e-8;

/// Small epsilon used for range / division guards.
const EPSILON: f64 = 1e-12;

/// Maximum number of samples used by [`estimate_white_noise_sigma`].
const MAX_SIGMA_SAMPLES: usize = 1_000_000;

// =============================================================================
// Configuration
// =============================================================================

/// Boundary handling for the frequency-domain inversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeblurBoundary {
    /// Mirror-extend the image before deconvolution and crop afterwards.
    /// Avoids the wrap-around ringing of a plain circular model. Default.
    #[default]
    Reflect,
    /// Treat the observation as circularly convolved (no padding).
    /// Only correct if the data really is periodic.
    Circular,
}

/// Configuration for BM3D-DEB deblurring.
///
/// `Default::default()` gives the reference settings; typically only `sigma`
/// (or nothing at all, if auto-estimation is wanted) needs to be set.
#[derive(Debug, Clone)]
pub struct Bm3dDeblurConfig<F: Bm3dFloat> {
    /// Standard deviation of the white noise in the *observation*, in input units.
    /// `<= 0` triggers automatic estimation. Default: 0.0 (auto).
    pub sigma: F,
    /// Stage-1 regularization factor (dimensionless). Default: 1.0
    pub reg_ri: F,
    /// Stage-2 Wiener regularization factor (dimensionless). Default: 5e-3
    pub reg_rwi: F,
    /// Hard thresholding coefficient for the first BM3D pass. Default: 2.7
    pub threshold: F,
    /// Block matching patch size. Default: 8
    pub patch_size: usize,
    /// Stride between reference patches. Default: 4
    pub step_size: usize,
    /// Search window size for block matching. Default: 24
    pub search_window: usize,
    /// Maximum similar patches per group. Default: 16
    pub max_matches: usize,
    /// Run the second (Wiener) stage. Disabling roughly halves the runtime at the
    /// cost of quality. Default: true
    pub wiener_stage: bool,
    /// Boundary handling. Default: [`DeblurBoundary::Reflect`]
    pub boundary: DeblurBoundary,
    /// Min-max normalize the input to `[0, 1]` before processing and restore the
    /// original range afterwards. The blur model is affine equivariant (the PSF sums
    /// to 1), so this is lossless with respect to the model. Default: true
    ///
    /// Independently of this flag, the image mean is removed before the frequency-domain
    /// inversion and added back afterwards, so a heavily regularized inversion decays
    /// towards the mean rather than towards whatever value normalization mapped to zero.
    pub normalize: bool,
}

impl<F: Bm3dFloat> Default for Bm3dDeblurConfig<F> {
    fn default() -> Self {
        Self {
            sigma: F::zero(),
            reg_ri: F::from_f64_c(DEFAULT_REG_RI),
            reg_rwi: F::from_f64_c(DEFAULT_REG_RWI),
            threshold: F::from_f64_c(DEFAULT_THRESHOLD),
            patch_size: DEFAULT_PATCH_SIZE,
            step_size: DEFAULT_STEP_SIZE,
            search_window: DEFAULT_SEARCH_WINDOW,
            max_matches: DEFAULT_MAX_MATCHES,
            wiener_stage: true,
            boundary: DeblurBoundary::default(),
            normalize: true,
        }
    }
}

impl<F: Bm3dFloat> Bm3dDeblurConfig<F> {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Validate the configuration parameters.
    pub fn validate(&self) -> Result<(), String> {
        if self.patch_size == 0 {
            return Err("patch_size must be > 0".to_string());
        }
        if self.step_size == 0 {
            return Err("step_size must be > 0".to_string());
        }
        if self.search_window == 0 {
            return Err("search_window must be > 0".to_string());
        }
        if self.max_matches == 0 {
            return Err("max_matches must be > 0".to_string());
        }
        if self.threshold < F::zero() {
            return Err("threshold must be >= 0".to_string());
        }
        if self.reg_ri <= F::zero() {
            return Err("reg_ri must be > 0".to_string());
        }
        if self.reg_rwi <= F::zero() {
            return Err("reg_rwi must be > 0".to_string());
        }
        Ok(())
    }
}

/// Result of [`bm3d_deblur_detailed`].
#[derive(Debug, Clone)]
pub struct Bm3dDeblurResult<F: Bm3dFloat> {
    /// Final estimate (stage 2 output, or stage 1 if `wiener_stage` is disabled).
    pub estimate: Array2<F>,
    /// Intermediate estimate from the regularized inverse stage.
    pub estimate_ri: Array2<F>,
    /// Noise standard deviation that was used, in input units.
    pub sigma: F,
}

// =============================================================================
// PSF Helpers
// =============================================================================

/// Normalize a PSF so that its entries sum to 1.
///
/// Fails for empty PSFs and for PSFs whose sum is (close to) zero.
pub fn normalize_psf<F: Bm3dFloat>(psf: ArrayView2<F>) -> Result<Array2<F>, String> {
    let (rows, cols) = psf.dim();
    if rows == 0 || cols == 0 {
        return Err("PSF must not be empty".to_string());
    }
    let sum = psf.iter().copied().fold(F::zero(), |a, b| a + b);
    if sum.abs() <= F::from_f64_c(EPSILON) {
        return Err("PSF sum must be non-zero".to_string());
    }
    Ok(psf.mapv(|x| x / sum))
}

/// Separable Gaussian PSF with the given standard deviations (in pixels).
///
/// The kernel is truncated at [`Bm3dFloat::GAUSSIAN_TRUNCATE`] sigmas and normalized.
pub fn gaussian_psf<F: Bm3dFloat>(sigma_rows: F, sigma_cols: F) -> Array2<F> {
    let kernel_rows = gaussian_kernel_1d::<F>(sigma_rows);
    let kernel_cols = gaussian_kernel_1d::<F>(sigma_cols);
    let mut psf = Array2::<F>::zeros((kernel_rows.len(), kernel_cols.len()));
    for (r, &kr) in kernel_rows.iter().enumerate() {
        for (c, &kc) in kernel_cols.iter().enumerate() {
            psf[[r, c]] = kr * kc;
        }
    }
    psf
}

/// Horizontal-only Gaussian PSF, shaped `(1, width)`.
///
/// Models detector / scintillator blur along the detector axis of a sinogram,
/// leaving the projection-angle axis untouched.
pub fn horizontal_gaussian_psf<F: Bm3dFloat>(sigma_cols: F) -> Array2<F> {
    let kernel = gaussian_kernel_1d::<F>(sigma_cols);
    let mut psf = Array2::<F>::zeros((1, kernel.len()));
    for (c, &k) in kernel.iter().enumerate() {
        psf[[0, c]] = k;
    }
    psf
}

/// Uniform (box / out-of-focus approximation) PSF of the given size.
pub fn boxcar_psf<F: Bm3dFloat>(rows: usize, cols: usize) -> Array2<F> {
    let rows = rows.max(1);
    let cols = cols.max(1);
    let value = F::one() / F::usize_as(rows * cols);
    Array2::from_elem((rows, cols), value)
}

/// Blur an image with a PSF using mirrored boundaries (direct convolution).
///
/// Uses the same centring convention as the deblurring model, so it can be used to
/// generate consistent test data:
/// `out[i, j] = sum_{r,c} psf[r, c] * image[i - (r - psf_rows/2), j - (c - psf_cols/2)]`.
///
/// Cost is `O(rows * cols * psf_rows * psf_cols)`, i.e. intended for small PSFs.
pub fn blur_with_psf<F: Bm3dFloat>(image: ArrayView2<F>, psf: ArrayView2<F>) -> Array2<F> {
    let (rows, cols) = image.dim();
    let (psf_rows, psf_cols) = psf.dim();
    if rows == 0 || cols == 0 || psf_rows == 0 || psf_cols == 0 {
        return image.to_owned();
    }
    let center_r = (psf_rows / 2) as isize;
    let center_c = (psf_cols / 2) as isize;

    Array2::from_shape_fn((rows, cols), |(i, j)| {
        let mut acc = F::zero();
        for r in 0..psf_rows {
            let src_r = reflect_index(i as isize - (r as isize - center_r), rows);
            for c in 0..psf_cols {
                let src_c = reflect_index(j as isize - (c as isize - center_c), cols);
                acc += psf[[r, c]] * image[[src_r, src_c]];
            }
        }
        acc
    })
}

fn gaussian_kernel_1d<F: Bm3dFloat>(sigma: F) -> Vec<F> {
    if sigma <= F::zero() {
        return vec![F::one()];
    }
    let radius = (F::GAUSSIAN_TRUNCATE * sigma)
        .ceil()
        .to_f64()
        .unwrap_or(1.0)
        .max(1.0) as usize;
    let len = 2 * radius + 1;
    let neg_half = F::from_f64_c(-0.5);
    let mut kernel = Vec::with_capacity(len);
    let mut sum = F::zero();
    for i in 0..len {
        let x = F::isize_as(i as isize - radius as isize);
        let normalized = x / sigma;
        let value = (neg_half * normalized * normalized).exp();
        sum += value;
        kernel.push(value);
    }
    if sum > F::zero() {
        for v in kernel.iter_mut() {
            *v = *v / sum;
        }
    }
    kernel
}

// =============================================================================
// Noise Estimation
// =============================================================================

/// Estimate the standard deviation of white noise using the robust MAD of the
/// diagonal Haar detail coefficients.
///
/// `d = (z[r,c] - z[r,c+1] - z[r+1,c] + z[r+1,c+1]) / 2` has variance `sigma^2` for
/// white noise, while being blind to smooth image content. This is a good fit for
/// deblurring inputs, where the signal has been low-pass filtered by the PSF and the
/// remaining high-frequency energy is dominated by the noise.
///
/// Note: this is *not* the same estimator as
/// [`crate::noise_estimation::estimate_noise_sigma`], which is tuned for vertical
/// streak noise in sinograms.
pub fn estimate_white_noise_sigma<F: Bm3dFloat>(image: ArrayView2<F>) -> F {
    let (rows, cols) = image.dim();
    if rows < 2 || cols < 2 {
        return F::zero();
    }
    let total = (rows - 1) * (cols - 1);
    let stride = (total / MAX_SIGMA_SAMPLES).max(1);
    let half = F::from_f64_c(0.5);

    let mut detail: Vec<F> = Vec::with_capacity(total / stride + 1);
    let mut index = 0usize;
    for r in 0..(rows - 1) {
        for c in 0..(cols - 1) {
            if index % stride == 0 {
                let d = (image[[r, c]] - image[[r, c + 1]] - image[[r + 1, c]]
                    + image[[r + 1, c + 1]])
                    * half;
                detail.push(d);
            }
            index += 1;
        }
    }

    let detail = Array1::from(detail);
    F::from_f64_c(estimate_robust_sigma(detail.view()))
}

// =============================================================================
// Internal Helpers
// =============================================================================

/// FFT plans for one full image size (rows x cols).
struct FourierPlans<F: Bm3dFloat> {
    fft_row: Arc<dyn Fft<F>>,
    fft_col: Arc<dyn Fft<F>>,
    ifft_row: Arc<dyn Fft<F>>,
    ifft_col: Arc<dyn Fft<F>>,
}

impl<F: Bm3dFloat> FourierPlans<F> {
    fn new(rows: usize, cols: usize) -> Self {
        let mut planner = FftPlanner::<F>::new();
        Self {
            fft_row: planner.plan_fft_forward(cols),
            fft_col: planner.plan_fft_forward(rows),
            ifft_row: planner.plan_fft_inverse(cols),
            ifft_col: planner.plan_fft_inverse(rows),
        }
    }
}

/// Forward FFT plans for the patch-sized (patch_size x patch_size) transform.
struct PatchPlans<F: Bm3dFloat> {
    fft_row: Arc<dyn Fft<F>>,
    fft_col: Arc<dyn Fft<F>>,
}

impl<F: Bm3dFloat> PatchPlans<F> {
    fn new(patch_size: usize) -> Self {
        let mut planner = FftPlanner::<F>::new();
        Self {
            fft_row: planner.plan_fft_forward(patch_size),
            fft_col: planner.plan_fft_forward(patch_size),
        }
    }
}

/// Everything that only depends on the (padded) image size and the PSF.
/// Reused across the slices of a stack.
struct DeblurWorkspace<F: Bm3dFloat> {
    fourier: FourierPlans<F>,
    patch: PatchPlans<F>,
    bm3d: Bm3dPlans<F>,
    /// Optical transfer function V(f) of the PSF on the padded grid.
    transfer: Array2<Complex<F>>,
}

impl<F: Bm3dFloat> DeblurWorkspace<F> {
    fn new(rows: usize, cols: usize, psf: ArrayView2<F>, config: &Bm3dDeblurConfig<F>) -> Self {
        let fourier = FourierPlans::new(rows, cols);
        let transfer = psf_transfer_function(psf, rows, cols, &fourier);
        Self {
            fourier,
            patch: PatchPlans::new(config.patch_size),
            bm3d: Bm3dPlans::new(config.patch_size, config.max_matches),
            transfer,
        }
    }
}

/// Mirror an out-of-range index back into `[0, n)` (period `2n - 2`, edge not repeated).
fn reflect_index(index: isize, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let n_i = n as isize;
    let period = 2 * n_i - 2;
    let mut k = index % period;
    if k < 0 {
        k += period;
    }
    if k >= n_i {
        k = period - k;
    }
    k as usize
}

/// Smallest 7-smooth integer `>= n`, i.e. a length rustfft handles efficiently.
fn next_fast_len(n: usize) -> usize {
    if n <= 4 {
        return n.max(1);
    }
    let mut candidate = n;
    loop {
        let mut rest = candidate;
        for p in [2usize, 3, 5, 7] {
            while rest % p == 0 {
                rest /= p;
            }
        }
        if rest == 1 {
            return candidate;
        }
        candidate += 1;
    }
}

/// Mirror-pad an image for deconvolution and round the size up to a fast FFT length.
///
/// Returns `(pad_rows, pad_cols, padded)`; the original image occupies
/// `padded[pad_rows .. pad_rows + rows, pad_cols .. pad_cols + cols]`.
fn pad_for_deconvolution<F: Bm3dFloat>(
    image: ArrayView2<F>,
    psf_dim: (usize, usize),
    patch_size: usize,
    boundary: DeblurBoundary,
) -> (usize, usize, Array2<F>) {
    let (rows, cols) = image.dim();
    if boundary == DeblurBoundary::Circular {
        return (0, 0, image.to_owned());
    }

    let pad_rows = (psf_dim.0 / 2 + 1).max(patch_size);
    let pad_cols = (psf_dim.1 / 2 + 1).max(patch_size);
    let target_rows = next_fast_len(rows + 2 * pad_rows);
    let target_cols = next_fast_len(cols + 2 * pad_cols);

    let padded = Array2::from_shape_fn((target_rows, target_cols), |(r, c)| {
        let src_r = reflect_index(r as isize - pad_rows as isize, rows);
        let src_c = reflect_index(c as isize - pad_cols as isize, cols);
        image[[src_r, src_c]]
    });

    (pad_rows, pad_cols, padded)
}

/// Optical transfer function of a PSF on a `rows x cols` grid.
///
/// The PSF is centred at `(psf_rows / 2, psf_cols / 2)` and circularly shifted so that
/// its centre lands on index `(0, 0)`, which makes the convolution phase-free.
fn psf_transfer_function<F: Bm3dFloat>(
    psf: ArrayView2<F>,
    rows: usize,
    cols: usize,
    plans: &FourierPlans<F>,
) -> Array2<Complex<F>> {
    let (psf_rows, psf_cols) = psf.dim();
    let center_r = psf_rows / 2;
    let center_c = psf_cols / 2;
    let mut big = Array2::<F>::zeros((rows, cols));

    for r in 0..psf_rows {
        // Wrap (r - center_r) into [0, rows); PSFs larger than the grid alias, which is
        // the correct circular behaviour.
        let dst_r = ((r + rows * (center_r / rows + 1)) - center_r) % rows;
        for c in 0..psf_cols {
            let dst_c = ((c + cols * (center_c / cols + 1)) - center_c) % cols;
            big[[dst_r, dst_c]] += psf[[r, c]];
        }
    }

    transforms::fft2d(big.view(), &plans.fft_row, &plans.fft_col)
}

/// Convert a frequency-domain noise PSD into per-2D-coefficient noise sigmas.
///
/// `noise_psd[f]` is the power spectral density in per-pixel units, i.e. the pixel
/// variance of the noise equals the mean of `noise_psd` over all frequencies.
///
/// For a stationary process the variance of the `patch_size x patch_size` DFT
/// coefficient `(r, c)` is
///
/// ```text
/// Var[r, c] = sum_{du,dv} (P - |du|)(P - |dv|) * R(du, dv) * exp(-2*pi*i*(r*du + c*dv)/P)
/// ```
///
/// with `R` the autocorrelation function (the inverse FFT of the PSD). The returned
/// value is `sqrt(Var / P^2)`, which is exactly the "equivalent white sigma" the BM3D
/// kernel expects: it computes `var = k * P^2 * sigma^2` for a group of `k` patches.
/// White noise of level `sigma` maps to a flat `sigma` map, as it should.
fn patch_noise_sigma<F: Bm3dFloat>(
    noise_psd: ArrayView2<F>,
    patch_size: usize,
    fourier: &FourierPlans<F>,
    patch: &PatchPlans<F>,
) -> Array2<F> {
    let (rows, cols) = noise_psd.dim();

    // 1. Autocorrelation function: R = IFFT(PSD), already normalized by 1/(rows*cols).
    let mut psd_complex = Array2::<Complex<F>>::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            psd_complex[[r, c]] = Complex::new(noise_psd[[r, c]], F::zero());
        }
    }
    let acf = transforms::ifft2d(&psd_complex, &fourier.ifft_row, &fourier.ifft_col);

    // 2. Apply the triangular (Bartlett) lag window of the finite patch window and fold
    //    the lags modulo patch_size.
    let p_i = patch_size as isize;
    let p_f = F::usize_as(patch_size);
    let mut folded = Array2::<F>::zeros((patch_size, patch_size));
    for du in -(p_i - 1)..p_i {
        let w_row = F::one() - F::isize_as(du.abs()) / p_f;
        let src_r = du.rem_euclid(rows as isize) as usize;
        let dst_r = du.rem_euclid(p_i) as usize;
        for dv in -(p_i - 1)..p_i {
            let w_col = F::one() - F::isize_as(dv.abs()) / p_f;
            let src_c = dv.rem_euclid(cols as isize) as usize;
            let dst_c = dv.rem_euclid(p_i) as usize;
            folded[[dst_r, dst_c]] += w_row * w_col * acf[[src_r, src_c]];
        }
    }

    // 3. Patch-sized DFT gives Var / P^2 per coefficient (imaginary part is ~0 because
    //    the windowed ACF is symmetric).
    let spectrum = transforms::fft2d(folded.view(), &patch.fft_row, &patch.fft_col);

    let mut mean_var = F::zero();
    for r in 0..patch_size {
        for c in 0..patch_size {
            mean_var += spectrum[[r, c]].re;
        }
    }
    mean_var = mean_var / F::usize_as(patch_size * patch_size);
    let floor = (mean_var * F::from_f64_c(PSD_FLOOR_RATIO)).max(F::from_f64_c(EPSILON * EPSILON));

    Array2::from_shape_fn((patch_size, patch_size), |(r, c)| {
        spectrum[[r, c]].re.max(floor).sqrt()
    })
}

/// Mean and (population) variance of an image.
fn mean_and_variance<F: Bm3dFloat>(image: ArrayView2<F>) -> (F, F) {
    let count = image.len();
    if count == 0 {
        return (F::zero(), F::zero());
    }
    let inv_n = F::one() / F::usize_as(count);
    let mean = image.iter().copied().fold(F::zero(), |a, b| a + b) * inv_n;
    let variance = image
        .iter()
        .copied()
        .fold(F::zero(), |acc, x| acc + (x - mean) * (x - mean))
        * inv_n;
    (mean, variance)
}

/// Multiply a spectrum by a frequency response, in place.
fn apply_filter<F: Bm3dFloat>(spectrum: &mut Array2<Complex<F>>, filter: &Array2<Complex<F>>) {
    let (rows, cols) = spectrum.dim();
    for r in 0..rows {
        for c in 0..cols {
            spectrum[[r, c]] = spectrum[[r, c]] * filter[[r, c]];
        }
    }
}

/// Core BM3D-DEB on an already padded / normalized image.
/// Returns `(final_estimate, ri_estimate)`, both on the padded grid.
fn deblur_padded<F: Bm3dFloat>(
    padded: ArrayView2<F>,
    sigma: F,
    config: &Bm3dDeblurConfig<F>,
    workspace: &DeblurWorkspace<F>,
) -> Result<(Array2<F>, Array2<F>), String> {
    let (rows, cols) = padded.dim();
    let transfer = &workspace.transfer;
    let fourier = &workspace.fourier;
    let sigma_sq = sigma * sigma;
    let min_reg = F::from_f64_c(MIN_REGULARIZATION);

    let (mean, variance) = mean_and_variance(padded);

    // Both inversions run on the mean-removed image, and the mean is added back before
    // each BM3D pass. The DC bin is the one frequency the PSF leaves untouched
    // (`V(0) == 1` because the PSF sums to 1), yet a regularized inverse still shrinks it
    // by `1 / (1 + lambda_RI)`. Whatever value maps to zero is therefore what the image
    // decays towards when the regularization is large. Centring on the mean makes that a
    // harmless loss of contrast; without it, the caller's normalization offset decides,
    // which for a min-max normalized chroma channel would drag the whole image towards
    // its most saturated extreme.
    let centered = padded.mapv(|x| x - mean);

    // Observation spectrum, reused by both stages.
    let observed_spectrum = transforms::fft2d(centered.view(), &fourier.fft_row, &fourier.fft_col);

    // --- Stage 1: regularized inversion -------------------------------------
    // Flat signal prior: use the observed variance minus the noise variance, with a
    // floor so that a noise-dominated input still yields a sane regularization.
    let signal_power = (variance - sigma_sq)
        .max(variance * F::from_f64_c(0.05))
        .max(F::from_f64_c(EPSILON));
    let lambda_ri = (config.reg_ri * sigma_sq / signal_power).max(min_reg);

    let mut ri_filter = Array2::<Complex<F>>::zeros((rows, cols));
    let mut psd_ri = Array2::<F>::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            let v = transfer[[r, c]];
            let filter = v.conj() / (v.norm_sqr() + lambda_ri);
            ri_filter[[r, c]] = filter;
            psd_ri[[r, c]] = sigma_sq * filter.norm_sqr();
        }
    }

    let mut ri_spectrum = observed_spectrum.clone();
    apply_filter(&mut ri_spectrum, &ri_filter);
    let mut z_ri = transforms::ifft2d(&ri_spectrum, &fourier.ifft_row, &fourier.ifft_col);
    z_ri.mapv_inplace(|x| x + mean);
    let coeff_sigma_ri =
        patch_noise_sigma(psd_ri.view(), config.patch_size, fourier, &workspace.patch);

    // Hard thresholding pass. The colored noise model lives in the 2D DFT basis, so the
    // Hadamard fast path must stay disabled.
    let dummy = Array2::<F>::zeros((1, 1));
    let hard_config = Bm3dKernelConfig {
        sigma_random: F::zero(),
        threshold: config.threshold,
        patch_size: config.patch_size,
        step_size: config.step_size,
        search_window: config.search_window,
        max_matches: config.max_matches,
        use_hadamard_fast_path: Some(false),
    };
    let y_ri = run_bm3d_step_colored_noise(
        z_ri.view(),
        z_ri.view(), // pilot = noisy input for the first pass
        Bm3dMode::HardThreshold,
        dummy.view(),
        dummy.view(),
        Some(coeff_sigma_ri.view()),
        &hard_config,
        &workspace.bm3d,
    )?;

    if !config.wiener_stage {
        return Ok((y_ri.clone(), y_ri));
    }

    // --- Stage 2: regularized Wiener inversion ------------------------------
    // The pilot is centred the same way as the observation, so the DC bin stays out of
    // both the signal PSD estimate and the inversion.
    let pilot_centered = y_ri.mapv(|x| x - mean);
    let pilot_spectrum =
        transforms::fft2d(pilot_centered.view(), &fourier.fft_row, &fourier.fft_col);
    let inv_n = F::one() / F::usize_as(rows * cols);

    let mut mean_signal_psd = F::zero();
    for r in 0..rows {
        for c in 0..cols {
            mean_signal_psd += pilot_spectrum[[r, c]].norm_sqr() * inv_n;
        }
    }
    mean_signal_psd = mean_signal_psd * inv_n;
    let rwi_reg = (config.reg_rwi * sigma_sq).max(min_reg * mean_signal_psd);

    let mut rwi_filter = Array2::<Complex<F>>::zeros((rows, cols));
    let mut psd_rwi = Array2::<F>::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            let v = transfer[[r, c]];
            // Pilot signal PSD in per-pixel units.
            let signal_psd = pilot_spectrum[[r, c]].norm_sqr() * inv_n;
            let denom = v.norm_sqr() * signal_psd + rwi_reg;
            let filter = if denom > F::zero() {
                v.conj() * signal_psd / denom
            } else {
                Complex::new(F::zero(), F::zero())
            };
            rwi_filter[[r, c]] = filter;
            psd_rwi[[r, c]] = sigma_sq * filter.norm_sqr();
        }
    }

    let mut rwi_spectrum = observed_spectrum;
    apply_filter(&mut rwi_spectrum, &rwi_filter);
    let mut z_rwi = transforms::ifft2d(&rwi_spectrum, &fourier.ifft_row, &fourier.ifft_col);
    z_rwi.mapv_inplace(|x| x + mean);
    let coeff_sigma_rwi =
        patch_noise_sigma(psd_rwi.view(), config.patch_size, fourier, &workspace.patch);

    let wiener_config = Bm3dKernelConfig {
        sigma_random: F::zero(),
        threshold: F::zero(), // not used for Wiener
        patch_size: config.patch_size,
        step_size: config.step_size,
        search_window: config.search_window,
        max_matches: config.max_matches,
        use_hadamard_fast_path: Some(false),
    };
    let y_final = run_bm3d_step_colored_noise(
        z_rwi.view(),
        y_ri.view(), // pilot = stage 1 estimate
        Bm3dMode::Wiener,
        dummy.view(),
        dummy.view(),
        Some(coeff_sigma_rwi.view()),
        &wiener_config,
        &workspace.bm3d,
    )?;

    Ok((y_final, y_ri))
}

/// Min-max normalization parameters: `work = (x - offset) / scale`.
fn normalization_params<F: Bm3dFloat>(image: ArrayView2<F>, enabled: bool) -> (F, F) {
    if !enabled {
        return (F::zero(), F::one());
    }
    let min = image
        .iter()
        .copied()
        .fold(F::infinity(), |a, b| if b < a { b } else { a });
    let max = image
        .iter()
        .copied()
        .fold(F::neg_infinity(), |a, b| if b > a { b } else { a });
    if !min.is_finite() || !max.is_finite() {
        return (F::zero(), F::one());
    }
    let range = max - min;
    if range > F::from_f64_c(EPSILON) {
        (min, range)
    } else {
        (min, F::one())
    }
}

// =============================================================================
// Public API
// =============================================================================

/// BM3D-DEB deblurring of a single 2D image.
///
/// # Arguments
///
/// * `observed` - Blurred and noisy observation (H x W)
/// * `psf` - Point spread function; normalized internally, must sum to a non-zero value
/// * `config` - Configuration, see [`Bm3dDeblurConfig`]
///
/// # Errors
///
/// Returns an error if the configuration is invalid, if the image is smaller than
/// `patch_size`, or if the PSF is empty / sums to zero.
pub fn bm3d_deblur<F: Bm3dFloat>(
    observed: ArrayView2<F>,
    psf: ArrayView2<F>,
    config: &Bm3dDeblurConfig<F>,
) -> Result<Array2<F>, String> {
    Ok(bm3d_deblur_detailed(observed, psf, config)?.estimate)
}

/// BM3D-DEB deblurring that also returns the intermediate estimate and the noise level.
///
/// Useful for diagnostics and for tuning `reg_ri` / `reg_rwi`.
pub fn bm3d_deblur_detailed<F: Bm3dFloat>(
    observed: ArrayView2<F>,
    psf: ArrayView2<F>,
    config: &Bm3dDeblurConfig<F>,
) -> Result<Bm3dDeblurResult<F>, String> {
    config.validate()?;
    let (rows, cols) = observed.dim();
    if rows < config.patch_size || cols < config.patch_size {
        return Err(format!(
            "Image size ({}, {}) is smaller than patch_size {}",
            rows, cols, config.patch_size
        ));
    }
    let psf = normalize_psf(psf)?;

    // 1. Normalize (the blur model is affine equivariant because the PSF sums to 1).
    let (offset, scale) = normalization_params(observed, config.normalize);
    let work = observed.mapv(|x| (x - offset) / scale);

    // 2. Noise level in working units.
    let sigma_work = if config.sigma > F::zero() {
        config.sigma / scale
    } else {
        estimate_white_noise_sigma(work.view())
    };

    // 3. Pad, deconvolve, crop.
    let (pad_rows, pad_cols, padded) =
        pad_for_deconvolution(work.view(), psf.dim(), config.patch_size, config.boundary);
    let (prows, pcols) = padded.dim();
    if prows < config.patch_size || pcols < config.patch_size {
        return Err(format!(
            "Padded size ({}, {}) is smaller than patch_size {}",
            prows, pcols, config.patch_size
        ));
    }

    let workspace = DeblurWorkspace::new(prows, pcols, psf.view(), config);
    let (estimate, estimate_ri) = deblur_padded(padded.view(), sigma_work, config, &workspace)?;

    let crop = |img: Array2<F>| -> Array2<F> {
        img.slice(s![pad_rows..pad_rows + rows, pad_cols..pad_cols + cols])
            .mapv(|x| x * scale + offset)
    };

    Ok(Bm3dDeblurResult {
        estimate: crop(estimate),
        estimate_ri: crop(estimate_ri),
        sigma: sigma_work * scale,
    })
}

/// BM3D-DEB deblurring of a 3D stack, slice by slice.
///
/// FFT plans and the transfer function are computed once and reused for every slice.
/// Normalization and (if enabled) noise estimation are done per slice.
///
/// An optional `progress_fn` callback is invoked after each slice with
/// `(completed, total)` counts. If the callback returns `Err`, the loop aborts and the
/// error is propagated.
pub fn bm3d_deblur_stack<F: Bm3dFloat>(
    observed: ArrayView3<F>,
    psf: ArrayView2<F>,
    config: &Bm3dDeblurConfig<F>,
    progress_fn: Option<&dyn Fn(usize, usize) -> Result<(), String>>,
) -> Result<Array3<F>, String> {
    config.validate()?;
    let (n, rows, cols) = observed.dim();
    if rows < config.patch_size || cols < config.patch_size {
        return Err(format!(
            "Image size ({}, {}) is smaller than patch_size {}",
            rows, cols, config.patch_size
        ));
    }
    let psf = normalize_psf(psf)?;

    let mut output = Array3::<F>::zeros((n, rows, cols));
    if n == 0 {
        return Ok(output);
    }

    // Padding geometry is identical for every slice, so build the workspace once.
    let first = observed.index_axis(Axis(0), 0);
    let (pad_rows, pad_cols, probe) =
        pad_for_deconvolution(first, psf.dim(), config.patch_size, config.boundary);
    let (prows, pcols) = probe.dim();
    if prows < config.patch_size || pcols < config.patch_size {
        return Err(format!(
            "Padded size ({}, {}) is smaller than patch_size {}",
            prows, pcols, config.patch_size
        ));
    }
    let workspace = DeblurWorkspace::new(prows, pcols, psf.view(), config);

    for i in 0..n {
        let slice = observed.index_axis(Axis(0), i);
        let (offset, scale) = normalization_params(slice, config.normalize);
        let work = slice.mapv(|x| (x - offset) / scale);
        let sigma_work = if config.sigma > F::zero() {
            config.sigma / scale
        } else {
            estimate_white_noise_sigma(work.view())
        };

        let (_pr, _pc, padded) =
            pad_for_deconvolution(work.view(), psf.dim(), config.patch_size, config.boundary);
        let (estimate, _estimate_ri) =
            deblur_padded(padded.view(), sigma_work, config, &workspace)?;

        let cropped = estimate
            .slice(s![pad_rows..pad_rows + rows, pad_cols..pad_cols + cols])
            .mapv(|x| x * scale + offset);
        output.slice_mut(s![i, .., ..]).assign(&cropped);

        if let Some(cb) = &progress_fn {
            cb(i + 1, n)?;
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    use rand_distr::{Distribution, Normal};

    fn test_config() -> Bm3dDeblurConfig<f32> {
        Bm3dDeblurConfig {
            patch_size: 8,
            step_size: 4,
            search_window: 16,
            max_matches: 8,
            ..Default::default()
        }
    }

    /// Piecewise-constant phantom with a smooth gradient background.
    fn phantom(rows: usize, cols: usize) -> Array2<f32> {
        Array2::from_shape_fn((rows, cols), |(r, c)| {
            let mut value = 0.2 + 0.3 * (c as f32 / cols as f32);
            if r > rows / 4 && r < 3 * rows / 4 && c > cols / 4 && c < 3 * cols / 4 {
                value += 0.4;
            }
            if r > rows / 2 && c > cols / 2 {
                value -= 0.2;
            }
            value
        })
    }

    fn mse(a: ArrayView2<f32>, b: ArrayView2<f32>) -> f32 {
        let n = a.len() as f32;
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            / n
    }

    #[test]
    fn test_next_fast_len() {
        assert_eq!(next_fast_len(1), 1);
        assert_eq!(next_fast_len(64), 64);
        assert_eq!(next_fast_len(100), 100); // 2^2 * 5^2
        assert_eq!(next_fast_len(101), 105); // 3 * 5 * 7
        assert!(next_fast_len(1000) >= 1000);
    }

    #[test]
    fn test_reflect_index() {
        assert_eq!(reflect_index(0, 5), 0);
        assert_eq!(reflect_index(4, 5), 4);
        assert_eq!(reflect_index(-1, 5), 1);
        assert_eq!(reflect_index(5, 5), 3);
        assert_eq!(reflect_index(-4, 5), 4);
        assert_eq!(reflect_index(3, 1), 0);
    }

    #[test]
    fn test_gaussian_psf_normalized() {
        let psf = gaussian_psf::<f64>(1.5, 2.5);
        let sum: f64 = psf.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "PSF sum was {}", sum);
        // 4-sigma truncation on both axes
        assert_eq!(psf.dim(), (13, 21));
    }

    #[test]
    fn test_boxcar_and_horizontal_psf() {
        let box_psf = boxcar_psf::<f32>(3, 3);
        assert_eq!(box_psf.dim(), (3, 3));
        assert!((box_psf.iter().sum::<f32>() - 1.0).abs() < 1e-6);

        let horizontal = horizontal_gaussian_psf::<f32>(1.0);
        assert_eq!(horizontal.dim().0, 1);
        assert!((horizontal.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_psf_rejects_zero_sum() {
        let psf = Array2::<f32>::zeros((3, 3));
        assert!(normalize_psf(psf.view()).is_err());
        let empty = Array2::<f32>::zeros((0, 3));
        assert!(normalize_psf(empty.view()).is_err());
    }

    #[test]
    fn test_delta_psf_is_identity_transfer() {
        let mut psf = Array2::<f64>::zeros((1, 1));
        psf[[0, 0]] = 1.0;
        let plans = FourierPlans::<f64>::new(16, 16);
        let transfer = psf_transfer_function(psf.view(), 16, 16, &plans);
        for value in transfer.iter() {
            assert!((value.re - 1.0).abs() < 1e-9);
            assert!(value.im.abs() < 1e-9);
        }
    }

    #[test]
    fn test_patch_noise_sigma_white_noise() {
        // A flat PSD of sigma^2 must map to a flat sigma map.
        let (rows, cols) = (32, 32);
        let sigma = 0.05f64;
        let psd = Array2::<f64>::from_elem((rows, cols), sigma * sigma);
        let fourier = FourierPlans::<f64>::new(rows, cols);
        let patch = PatchPlans::<f64>::new(8);

        let coeff_sigma = patch_noise_sigma(psd.view(), 8, &fourier, &patch);
        assert_eq!(coeff_sigma.dim(), (8, 8));
        for value in coeff_sigma.iter() {
            assert!(
                (value - sigma).abs() < 1e-9,
                "expected {}, got {}",
                sigma,
                value
            );
        }
    }

    #[test]
    fn test_patch_noise_sigma_lowpass_is_anisotropic() {
        // Noise that only lives at low vertical frequencies must give larger sigmas for
        // the low-frequency coefficients than for the high-frequency ones.
        let (rows, cols) = (32, 32);
        let mut psd = Array2::<f64>::zeros((rows, cols));
        for r in 0..rows {
            let dist = r.min(rows - r) as f64;
            let weight = (-0.5 * (dist / 2.0) * (dist / 2.0)).exp();
            for c in 0..cols {
                psd[[r, c]] = weight;
            }
        }
        let fourier = FourierPlans::<f64>::new(rows, cols);
        let patch = PatchPlans::<f64>::new(8);
        let coeff_sigma = patch_noise_sigma(psd.view(), 8, &fourier, &patch);

        assert!(coeff_sigma[[0, 0]] > coeff_sigma[[4, 0]]);
        assert!(coeff_sigma[[0, 0]] > 0.0);
        for value in coeff_sigma.iter() {
            assert!(value.is_finite() && *value >= 0.0);
        }
    }

    #[test]
    fn test_estimate_white_noise_sigma() {
        let (rows, cols) = (128, 128);
        let sigma_true = 0.03f32;
        let mut rng = StdRng::seed_from_u64(7);
        let normal = Normal::new(0.0f32, sigma_true).unwrap();

        let clean = phantom(rows, cols);
        let noisy = clean.mapv(|x| x + normal.sample(&mut rng));

        let estimated = estimate_white_noise_sigma(noisy.view());
        let error = (estimated - sigma_true).abs() / sigma_true;
        assert!(
            error < 0.20,
            "sigma estimate {} vs {}",
            estimated,
            sigma_true
        );
    }

    #[test]
    fn test_blur_with_psf_preserves_mean() {
        let image = phantom(64, 64);
        let psf = gaussian_psf::<f32>(1.5, 1.5);
        let blurred = blur_with_psf(image.view(), psf.view());

        let mean_before = image.iter().sum::<f32>() / image.len() as f32;
        let mean_after = blurred.iter().sum::<f32>() / blurred.len() as f32;
        assert!((mean_before - mean_after).abs() < 5e-3);
    }

    #[test]
    fn test_deblur_improves_mse() {
        let (rows, cols) = (96, 96);
        let clean = phantom(rows, cols);
        let psf = gaussian_psf::<f32>(1.2, 1.2);
        let blurred = blur_with_psf(clean.view(), psf.view());

        let sigma_true = 0.01f32;
        let mut rng = StdRng::seed_from_u64(1234);
        let normal = Normal::new(0.0f32, sigma_true).unwrap();
        let observed = blurred.mapv(|x| x + normal.sample(&mut rng));

        let mut config = test_config();
        config.sigma = sigma_true;

        let restored = bm3d_deblur(observed.view(), psf.view(), &config).unwrap();
        assert_eq!(restored.dim(), (rows, cols));

        let mse_observed = mse(observed.view(), clean.view());
        let mse_restored = mse(restored.view(), clean.view());
        assert!(
            mse_restored < mse_observed,
            "deblurring did not improve MSE: {} -> {}",
            mse_observed,
            mse_restored
        );
        assert!(restored.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_deblur_auto_sigma_and_detailed_output() {
        let (rows, cols) = (64, 64);
        let clean = phantom(rows, cols);
        let psf = gaussian_psf::<f32>(1.0, 1.0);
        let blurred = blur_with_psf(clean.view(), psf.view());

        let sigma_true = 0.02f32;
        let mut rng = StdRng::seed_from_u64(99);
        let normal = Normal::new(0.0f32, sigma_true).unwrap();
        let observed = blurred.mapv(|x| x + normal.sample(&mut rng));

        let config = test_config(); // sigma = 0.0 -> auto estimation
        let result = bm3d_deblur_detailed(observed.view(), psf.view(), &config).unwrap();

        assert_eq!(result.estimate.dim(), (rows, cols));
        assert_eq!(result.estimate_ri.dim(), (rows, cols));
        let error = (result.sigma - sigma_true).abs() / sigma_true;
        assert!(
            error < 0.35,
            "auto sigma {} vs {}",
            result.sigma,
            sigma_true
        );
    }

    #[test]
    fn test_deblur_delta_psf_preserves_level() {
        // With a delta PSF, BM3D-DEB degenerates to plain (colored-noise) denoising.
        let (rows, cols) = (64, 64);
        let clean = phantom(rows, cols);
        let sigma_true = 0.02f32;
        let mut rng = StdRng::seed_from_u64(5);
        let normal = Normal::new(0.0f32, sigma_true).unwrap();
        let observed = clean.mapv(|x| x + normal.sample(&mut rng));

        let mut psf = Array2::<f32>::zeros((1, 1));
        psf[[0, 0]] = 1.0;

        let mut config = test_config();
        config.sigma = sigma_true;
        let restored = bm3d_deblur(observed.view(), psf.view(), &config).unwrap();

        let mse_observed = mse(observed.view(), clean.view());
        let mse_restored = mse(restored.view(), clean.view());
        assert!(
            mse_restored < mse_observed,
            "denoising did not improve MSE: {} -> {}",
            mse_observed,
            mse_restored
        );
    }

    #[test]
    fn test_deblur_no_wiener_stage() {
        let (rows, cols) = (48, 48);
        let clean = phantom(rows, cols);
        let psf = gaussian_psf::<f32>(1.0, 1.0);
        let observed = blur_with_psf(clean.view(), psf.view());

        let mut config = test_config();
        config.sigma = 0.01;
        config.wiener_stage = false;

        let result = bm3d_deblur_detailed(observed.view(), psf.view(), &config).unwrap();
        assert_eq!(result.estimate.dim(), (rows, cols));
        // Without the second stage both outputs are the stage-1 estimate.
        assert_eq!(result.estimate, result.estimate_ri);
    }

    #[test]
    fn test_heavy_regularization_decays_to_mean_not_minimum() {
        // Regression test: with a noise level comparable to the whole data range the
        // inversion collapses. The output must then fall back to the image mean. Falling
        // back to the minimum instead would, for a chroma channel, read as a hue shift.
        let (rows, cols) = (48, 48);
        let image = Array2::<f32>::from_shape_fn((rows, cols), |(r, c)| {
            // Offset range: min is far from zero and far from the mean.
            -0.4 + 0.5 * ((r + c) as f32 / (rows + cols) as f32)
        });
        let mean = image.iter().sum::<f32>() / image.len() as f32;
        let min = image.iter().copied().fold(f32::INFINITY, f32::min);

        let psf = gaussian_psf::<f32>(1.5, 1.5);
        let mut config = test_config();
        config.sigma = 10.0; // absurd relative to the data range

        let restored = bm3d_deblur(image.view(), psf.view(), &config).unwrap();
        let restored_mean = restored.iter().sum::<f32>() / restored.len() as f32;

        assert!(
            (restored_mean - mean).abs() < 0.05,
            "expected decay towards the mean {}, got {}",
            mean,
            restored_mean
        );
        assert!(
            (restored_mean - min).abs() > 0.1,
            "output collapsed towards the minimum {}",
            min
        );
    }

    #[test]
    fn test_deblur_stack() {
        let (n, rows, cols) = (3, 48, 48);
        let clean = phantom(rows, cols);
        let psf = gaussian_psf::<f32>(1.0, 1.0);
        let blurred = blur_with_psf(clean.view(), psf.view());

        let mut rng = StdRng::seed_from_u64(31);
        let normal = Normal::new(0.0f32, 0.01f32).unwrap();
        let mut stack = Array3::<f32>::zeros((n, rows, cols));
        for i in 0..n {
            let noisy = blurred.mapv(|x| x + normal.sample(&mut rng));
            stack.slice_mut(s![i, .., ..]).assign(&noisy);
        }

        let mut config = test_config();
        config.sigma = 0.01;

        let seen = std::cell::Cell::new(0usize);
        let progress = |done: usize, total: usize| -> Result<(), String> {
            assert_eq!(total, n);
            assert_eq!(done, seen.get() + 1);
            seen.set(done);
            Ok(())
        };
        let restored = bm3d_deblur_stack(stack.view(), psf.view(), &config, Some(&progress))
            .expect("stack deblurring failed");

        assert_eq!(seen.get(), n);
        assert_eq!(restored.dim(), (n, rows, cols));
        assert!(restored.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_invalid_inputs_are_rejected() {
        let image = Array2::<f32>::zeros((4, 4));
        let psf = gaussian_psf::<f32>(1.0, 1.0);
        let config = test_config();
        // Image smaller than patch_size
        assert!(bm3d_deblur(image.view(), psf.view(), &config).is_err());

        let image = Array2::<f32>::zeros((32, 32));
        let mut bad_config = test_config();
        bad_config.step_size = 0;
        assert!(bm3d_deblur(image.view(), psf.view(), &bad_config).is_err());
    }

    #[test]
    fn test_f64_support() {
        let (rows, cols) = (48, 48);
        let clean = Array2::<f64>::from_shape_fn((rows, cols), |(r, c)| {
            if r > 12 && r < 36 && c > 12 && c < 36 {
                1.0
            } else {
                0.0
            }
        });
        let psf = gaussian_psf::<f64>(1.0, 1.0);
        let observed = blur_with_psf(clean.view(), psf.view());

        let config = Bm3dDeblurConfig::<f64> {
            sigma: 1e-3,
            patch_size: 8,
            step_size: 4,
            search_window: 16,
            max_matches: 8,
            ..Default::default()
        };
        let restored = bm3d_deblur(observed.view(), psf.view(), &config).unwrap();
        assert_eq!(restored.dim(), (rows, cols));
        assert!(restored.iter().all(|x| x.is_finite()));
    }
}
