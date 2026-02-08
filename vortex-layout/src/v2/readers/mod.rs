// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_dtype::DecimalType;

pub mod chunked;
pub mod constant;
pub mod dict;
pub mod flat;
pub mod scalar_fn;
pub mod struct_;
pub mod zoned;

/// Estimates the decoded in-memory byte size for `n` elements of the given type.
///
/// For fixed-width types this is exact. For variable-width types (strings, binary, lists)
/// a heuristic estimate is used.
fn estimated_decoded_bytes(dtype: &DType, n: usize) -> usize {
    const VARIABLE_ELEMENT_ESTIMATE: usize = 64;
    match dtype {
        DType::Null => 0,
        DType::Bool(_) => n,
        DType::Primitive(ptype, _) => n * ptype.byte_width(),
        DType::Decimal(decimal, _) => {
            n * DecimalType::smallest_decimal_value_type(decimal).byte_width()
        }
        DType::Utf8(_) | DType::Binary(_) => n * VARIABLE_ELEMENT_ESTIMATE,
        DType::List(elem_dtype, _) => n * 10 * estimated_decoded_bytes(elem_dtype, 1),
        DType::FixedSizeList(elem_dtype, list_size, _) => {
            n * estimated_decoded_bytes(elem_dtype, *list_size as usize)
        }
        DType::Struct(fields, _) => fields
            .fields()
            .map(|f| estimated_decoded_bytes(&f, n))
            .sum(),
        DType::Extension(ext) => estimated_decoded_bytes(ext.storage_dtype(), n),
    }
}
