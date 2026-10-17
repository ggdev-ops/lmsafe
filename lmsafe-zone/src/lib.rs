//! lmsafe-zone — the safe zone: where models and media live, and the
//! admission gate that decides what may be loaded.
//!
//! Depends on `sysreflect-core` (what is the system doing?) and
//! `sysreflect-model` (what is this file?). Neither depends on us.

pub mod admission;
pub mod storage;

pub use admission::{
    AdmissionReport, Verdict, assess_admission, assess_model_path, assess_with_monitor,
    model_is_safe_to_load,
};
