// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ConstantVTable;
use crate::arrow::Datum;
use crate::arrow::from_arrow_array_with_len;
use crate::expr::exprs::operators::Operator;
use crate::scalar::NumericOperator;

/// Execute a numeric operation between two arrays.
///
/// This is the entry point for numeric operations from the binary expression.
/// Handles constant-constant directly, otherwise falls back to Arrow.
pub(crate) fn execute_numeric(
    lhs: &dyn Array,
    rhs: &dyn Array,
    op: Operator,
) -> VortexResult<ArrayRef> {
    if let Some(result) = constant_numeric(lhs, rhs, op)? {
        return Ok(result);
    }
    arrow_numeric(lhs, rhs, op)
}

fn constant_numeric(
    lhs: &dyn Array,
    rhs: &dyn Array,
    op: Operator,
) -> VortexResult<Option<ArrayRef>> {
    let (Some(lhs), Some(rhs)) = (
        lhs.as_opt::<ConstantVTable>(),
        rhs.as_opt::<ConstantVTable>(),
    ) else {
        return Ok(None);
    };

    let scalar_op = NumericOperator::try_from(op)?;

    Ok(Some(
        ConstantArray::new(
            lhs.scalar()
                .as_primitive()
                .checked_binary_numeric(&rhs.scalar().as_primitive(), scalar_op)
                .ok_or_else(|| vortex_err!("numeric overflow"))?,
            lhs.len(),
        )
        .into_array(),
    ))
}

/// Implementation of numeric operations using the Arrow crate.
fn arrow_numeric(lhs: &dyn Array, rhs: &dyn Array, operator: Operator) -> VortexResult<ArrayRef> {
    let nullable = lhs.dtype().is_nullable() || rhs.dtype().is_nullable();
    let len = lhs.len();

    let left = Datum::try_new(lhs)?;
    let right = Datum::try_new_with_target_datatype(rhs, left.data_type())?;

    let array = match operator {
        Operator::Add => arrow_arith::numeric::add(&left, &right)?,
        Operator::Sub => arrow_arith::numeric::sub(&left, &right)?,
        Operator::Mul => arrow_arith::numeric::mul(&left, &right)?,
        Operator::Div => arrow_arith::numeric::div(&left, &right)?,
        other => vortex_bail!("unsupported numeric operator: {}", other),
    };

    from_arrow_array_with_len(array.as_ref(), len, nullable)
}
