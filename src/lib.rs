// Include the XGBoost C API bindings
mod sys;

mod error;
pub use crate::error::{XGBoostError, XGBoostResult};

mod model;
pub use crate::model::Booster;

#[cfg(feature = "polars")]
mod polars_ext;
#[cfg(feature = "polars")]
pub use crate::polars_ext::BoosterPolarsExt;

// Re-export prediction option constants for convenience
pub mod predict_option {
    /// Normal prediction, output is the transformed probability
    pub const OUTPUT_MARGIN: u32 = 0x01;
    /// Output the untransformed margin value
    pub const PRED_LEAF: u32 = 0x02;
    /// Output the leaf index of trees
    pub const PRED_CONTRIBS: u32 = 0x04;
    /// Output feature contributions (SHAP values)
    pub const PRED_APPROX_CONTRIBS: u32 = 0x08;
    /// Output feature interaction contributions
    pub const PRED_INTERACTIONS: u32 = 0x10;
}
