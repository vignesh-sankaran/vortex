// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::expr::Operator;
use crate::expr::execute_numeric;
use crate::scalar::NumericOperator;
use crate::scalar::Scalar;

/// Point-wise add two numeric arrays.
///
/// Errs at runtime if the sum would overflow or underflow.
///
/// The result is null at any index that either input is null.
#[deprecated(note = "Use checked_add expression instead")]
pub fn add(lhs: &dyn Array, rhs: &dyn Array) -> VortexResult<ArrayRef> {
    #[expect(deprecated)]
    numeric(lhs, rhs, NumericOperator::Add)
}

/// Point-wise add a scalar value to this array on the right-hand-side.
#[deprecated(note = "Use checked_add expression instead")]
pub fn add_scalar(lhs: &dyn Array, rhs: Scalar) -> VortexResult<ArrayRef> {
    #[expect(deprecated)]
    numeric(
        lhs,
        &ConstantArray::new(rhs, lhs.len()).into_array(),
        NumericOperator::Add,
    )
}

/// Point-wise subtract two numeric arrays.
#[deprecated(note = "Use checked_sub expression instead")]
pub fn sub(lhs: &dyn Array, rhs: &dyn Array) -> VortexResult<ArrayRef> {
    #[expect(deprecated)]
    numeric(lhs, rhs, NumericOperator::Sub)
}

/// Point-wise subtract a scalar value from this array on the right-hand-side.
#[deprecated(note = "Use checked_sub expression instead")]
pub fn sub_scalar(lhs: &dyn Array, rhs: Scalar) -> VortexResult<ArrayRef> {
    #[expect(deprecated)]
    numeric(
        lhs,
        &ConstantArray::new(rhs, lhs.len()).into_array(),
        NumericOperator::Sub,
    )
}

/// Point-wise multiply two numeric arrays.
#[deprecated(note = "Use checked_mul expression instead")]
pub fn mul(lhs: &dyn Array, rhs: &dyn Array) -> VortexResult<ArrayRef> {
    #[expect(deprecated)]
    numeric(lhs, rhs, NumericOperator::Mul)
}

/// Point-wise multiply a scalar value into this array on the right-hand-side.
#[deprecated(note = "Use checked_mul expression instead")]
pub fn mul_scalar(lhs: &dyn Array, rhs: Scalar) -> VortexResult<ArrayRef> {
    #[expect(deprecated)]
    numeric(
        lhs,
        &ConstantArray::new(rhs, lhs.len()).into_array(),
        NumericOperator::Mul,
    )
}

/// Point-wise divide two numeric arrays.
#[deprecated(note = "Use checked_div expression instead")]
pub fn div(lhs: &dyn Array, rhs: &dyn Array) -> VortexResult<ArrayRef> {
    #[expect(deprecated)]
    numeric(lhs, rhs, NumericOperator::Div)
}

/// Point-wise divide a scalar value into this array on the right-hand-side.
#[deprecated(note = "Use checked_div expression instead")]
pub fn div_scalar(lhs: &dyn Array, rhs: Scalar) -> VortexResult<ArrayRef> {
    #[expect(deprecated)]
    numeric(
        lhs,
        &ConstantArray::new(rhs, lhs.len()).into_array(),
        NumericOperator::Div,
    )
}

/// Point-wise numeric operation between two arrays of the same type and length.
#[deprecated(note = "Use checked_add/checked_sub/checked_mul/checked_div expressions instead")]
pub fn numeric(lhs: &dyn Array, rhs: &dyn Array, op: NumericOperator) -> VortexResult<ArrayRef> {
    execute_numeric(lhs, rhs, Operator::from(op))
}

#[cfg(test)]
mod test {
    use vortex_buffer::buffer;

    use crate::IntoArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;

    #[expect(deprecated)]
    #[test]
    fn test_scalar_subtract_unsigned() {
        let values = buffer![1u16, 2, 3].into_array();
        let result = super::sub_scalar(&values, 1u16.into()).unwrap();
        assert_arrays_eq!(result, PrimitiveArray::from_iter([0u16, 1, 2]));
    }

    #[expect(deprecated)]
    #[test]
    fn test_scalar_subtract_signed() {
        let values = buffer![1i64, 2, 3].into_array();
        let result = super::sub_scalar(&values, (-1i64).into()).unwrap();
        assert_arrays_eq!(result, PrimitiveArray::from_iter([2i64, 3, 4]));
    }

    #[expect(deprecated)]
    #[test]
    fn test_scalar_subtract_nullable() {
        let values = PrimitiveArray::from_option_iter([Some(1u16), Some(2), None, Some(3)]);
        let result = super::sub_scalar(values.as_ref(), Some(1u16).into()).unwrap();
        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([Some(0u16), Some(1), None, Some(2)])
        );
    }

    #[expect(deprecated)]
    #[test]
    fn test_scalar_subtract_float() {
        let values = buffer![1.0f64, 2.0, 3.0].into_array();
        let to_subtract = -1f64;
        let result = super::sub_scalar(&values, to_subtract.into()).unwrap();
        assert_arrays_eq!(result, PrimitiveArray::from_iter([2.0f64, 3.0, 4.0]));
    }

    #[expect(deprecated)]
    #[test]
    fn test_scalar_subtract_float_underflow_is_ok() {
        let values = buffer![f32::MIN, 2.0, 3.0].into_array();
        let _results = super::sub_scalar(&values, 1.0f32.into()).unwrap();
        let _results = super::sub_scalar(&values, f32::MAX.into()).unwrap();
    }
}
