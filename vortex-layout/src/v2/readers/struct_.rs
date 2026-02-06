// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::ops::Range;
use std::sync::Arc;

use futures::future::BoxFuture;
use futures::future::try_join_all;
use futures::try_join;
use itertools::Itertools;
use moka::future::FutureExt;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::arrays::StructArray;
use vortex_array::expr::Expression;
use vortex_array::expr::GetItem;
use vortex_array::expr::Literal;
use vortex_array::expr::Root;
use vortex_array::validity::Validity;
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::v2::reader::Reader;
use crate::v2::reader::ReaderRef;
use crate::v2::reader::ReaderStream;
use crate::v2::reader::ReaderStreamRef;
use crate::v2::readers::constant::ConstantReader;
use crate::v2::readers::scalar_fn::ScalarFnReader;

/// A reader over a struct with named fields.
pub struct StructReader {
    row_count: u64,
    dtype: DType,
    validity: Option<ReaderRef>,
    fields: Vec<ReaderRef>,
}

impl StructReader {
    pub fn new(
        row_count: u64,
        dtype: DType,
        validity: Option<ReaderRef>,
        fields: Vec<ReaderRef>,
    ) -> Self {
        Self {
            row_count,
            dtype,
            validity,
            fields,
        }
    }

    /// Recursively resolve an expression through this struct, extracting fields where possible.
    fn resolve_expr(&self, expr: &Expression) -> VortexResult<ReaderRef> {
        // Root references this struct.
        if expr.is::<Root>() {
            return Ok(Arc::new(Self {
                row_count: self.row_count,
                dtype: self.dtype.clone(),
                validity: self.validity.clone(),
                fields: self.fields.clone(),
            }));
        }

        // Literals become constant readers.
        if let Some(scalar) = expr.as_opt::<Literal>() {
            return Ok(Arc::new(ConstantReader::new(
                scalar.clone(),
                self.row_count,
            )));
        }

        // Recursively resolve all children through this struct.
        let resolved_children: Vec<ReaderRef> = expr
            .children()
            .iter()
            .map(|child| self.resolve_expr(child))
            .try_collect()?;

        // If this is GetItem and the resolved child is a StructReader, extract the field directly.
        if let Some(field_name) = expr.as_opt::<GetItem>() {
            debug_assert_eq!(resolved_children.len(), 1);
            let child_reader = &resolved_children[0];
            if let Some(struct_reader) = child_reader.as_any().downcast_ref::<StructReader>() {
                let struct_fields = struct_reader
                    .dtype
                    .as_struct_fields_opt()
                    .ok_or_else(|| vortex_error::vortex_err!("Expected struct dtype"))?;
                let field_idx = struct_fields.find(field_name).ok_or_else(|| {
                    vortex_error::vortex_err!("Field '{}' not found in struct", field_name)
                })?;
                return Ok(struct_reader.fields[field_idx].clone());
            }
            // Otherwise, the child is some other reader (e.g., ChunkedReader). Delegate apply.
            return child_reader.apply(expr);
        }

        // For any other scalar function, wrap the resolved children.
        Ok(Arc::new(ScalarFnReader::try_new(
            expr.scalar_fn().clone(),
            resolved_children,
            self.row_count,
        )?))
    }
}

impl Reader for StructReader {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn row_count(&self) -> u64 {
        self.row_count
    }

    fn apply(&self, expression: &Expression) -> VortexResult<ReaderRef> {
        self.resolve_expr(expression)
    }

    fn execute(&self, row_range: Range<u64>) -> VortexResult<ReaderStreamRef> {
        let validity_stream = self
            .validity
            .as_ref()
            .map(|v| v.execute(row_range.clone()))
            .transpose()?;
        let field_streams = self
            .fields
            .iter()
            .map(|field| field.execute(row_range.clone()))
            .collect::<VortexResult<Vec<_>>>()?;

        Ok(Box::new(StructReaderStream {
            dtype: self.dtype.clone(),
            validity: validity_stream,
            fields: field_streams,
        }))
    }
}

struct StructReaderStream {
    dtype: DType,
    validity: Option<ReaderStreamRef>,
    fields: Vec<ReaderStreamRef>,
}

impl ReaderStream for StructReaderStream {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn next_chunk_len(&self) -> Option<usize> {
        let field_min = self
            .fields
            .iter()
            .map(|s| s.next_chunk_len())
            .min()
            .flatten();
        match (&self.validity, field_min) {
            (Some(v), Some(f)) => v.next_chunk_len().map(|vl| vl.min(f)),
            (Some(v), None) => v.next_chunk_len(),
            (None, f) => f,
        }
    }

    fn skip(&mut self, n: usize) {
        if let Some(validity) = &mut self.validity {
            validity.skip(n);
        }
        for field in &mut self.fields {
            field.skip(n);
        }
    }

    fn next_chunk(
        &mut self,
        mask: MaskFuture,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ArrayRef>>> {
        let struct_fields = self
            .dtype
            .as_struct_fields_opt()
            .ok_or_else(|| vortex_error::vortex_err!("Expected struct dtype"))?
            .clone();
        let nullability = self.dtype.nullability();
        let validity_fut = self
            .validity
            .as_mut()
            .map(|v| v.next_chunk(mask.clone()))
            .transpose()?;
        let fields = self
            .fields
            .iter_mut()
            .map(|s| s.next_chunk(mask.clone()))
            .collect::<VortexResult<Vec<_>>>()?;

        Ok(async move {
            let fields = try_join_all(fields);
            let (fields, mask) = try_join!(fields, mask)?;
            let validity = if let Some(validity_fut) = validity_fut {
                let validity_array = validity_fut.await?;
                Validity::Array(validity_array)
            } else {
                nullability.into()
            };
            Ok(
                StructArray::try_new_with_dtype(
                    fields,
                    struct_fields,
                    mask.true_count(),
                    validity,
                )?
                .into_array(),
            )
        }
        .boxed())
    }
}
