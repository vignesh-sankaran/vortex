// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_scalar::Scalar;

use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::expr::Arity;
use crate::expr::ChildName;
use crate::expr::ExecutionArgs;
use crate::expr::ExprId;
use crate::expr::Expression;
use crate::expr::SimplifyCtx;
use crate::expr::VTable;
use crate::expr::VTableExt;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProvider;

/// Creates a new expression that returns a minimum bound of its input.
pub fn statistic(stat: Stat, child: Expression) -> Expression {
    Statistic.new_expr(stat, vec![child])
}

pub struct Statistic;

impl VTable for Statistic {
    type Options = Stat;

    fn id(&self) -> ExprId {
        ExprId::from("statistic")
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &Self::Options, _child_idx: usize) -> ChildName {
        ChildName::from("input")
    }

    fn return_dtype(&self, stat: &Stat, arg_dtypes: &[DType]) -> VortexResult<DType> {
        stat.dtype(&arg_dtypes[0])
            .ok_or_else(|| {
                vortex_err!(
                    "statistic {:?} not supported for dtype {:?}",
                    stat,
                    arg_dtypes[0]
                )
            })
            // We make all statistics types nullable in case there is no reduction rule to handle
            // the statistic expression.
            .map(|dt| dt.as_nullable())
    }

    fn execute(&self, stat: &Stat, args: ExecutionArgs) -> VortexResult<ArrayRef> {
        // FIXME(ngates): we should implement this as a reduction rule instead?
        let Some(stat_dtype) = stat.dtype(args.inputs[0].dtype()) else {
            vortex_bail!(
                "Statistic {:?} not supported for dtype {:?}",
                stat,
                args.inputs[0].dtype()
            );
        };

        Ok(match args.inputs[0].statistics().get(*stat) {
            // TODO(ngates): do we care about precision here? Possibly we should configure in the
            //  options of the expression.
            Some(Precision::Exact(v)) => {
                // We have an exact value for the statistic; so we return a constant array
                // with that value.
                ConstantArray::new(v, args.row_count).into_array()
            }
            None | Some(Precision::Inexact(_)) => {
                ConstantArray::new(Scalar::null(stat_dtype), args.row_count).into_array()
            }
        })
    }

    fn simplify(
        &self,
        _options: &Self::Options,
        _expr: &Expression,
        _ctx: &dyn SimplifyCtx,
    ) -> VortexResult<Option<Expression>> {
        // FIXME(ngates): we really want to implement a reduction rule for all arrays? But it's an array.
        //  And it's a reduction rule. How do we do this without reduce_parent on everything..?
        Ok(None)
    }
}
