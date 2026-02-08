// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::ops::Range;
use std::sync::Arc;

use futures::future::try_join_all;
use itertools::Itertools;
use termtree::Tree;
use vortex_array::ArrayFuture;
use vortex_array::IntoArray;
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

        let num_fields = field_streams.len();
        Ok(Box::new(StructReaderStream {
            dtype: self.dtype.clone(),
            validity: validity_stream,
            fields: field_streams,
            validity_buffer: None,
            field_buffers: vec![None; num_fields],
        }))
    }

    fn display_tree(&self) -> Tree<String> {
        let label = format!("Struct({}, rows={})", self.dtype, self.row_count);
        let mut tree = Tree::new(label);

        // Add field children with names from the struct dtype.
        if let Some(struct_fields) = self.dtype.as_struct_fields_opt() {
            for (name, field_reader) in struct_fields.names().iter().zip(self.fields.iter()) {
                let child = field_reader.display_tree();
                tree.push(Tree::new(format!("{}: {}", name, child.root)).with_leaves(child.leaves));
            }
        } else {
            for (i, field_reader) in self.fields.iter().enumerate() {
                let child = field_reader.display_tree();
                tree.push(Tree::new(format!("[{}]: {}", i, child.root)).with_leaves(child.leaves));
            }
        }

        // Add validity child if present.
        if let Some(validity) = &self.validity {
            let child = validity.display_tree();
            tree.push(Tree::new(format!("validity: {}", child.root)).with_leaves(child.leaves));
        }

        tree
    }
}

struct StructReaderStream {
    dtype: DType,
    validity: Option<ReaderStreamRef>,
    fields: Vec<ReaderStreamRef>,
    validity_buffer: Option<ArrayFuture>,
    field_buffers: Vec<Option<ArrayFuture>>,
}

impl StructReaderStream {
    /// Get the next ArrayFuture for a child stream, taking from the buffer first.
    fn next_for_child(
        stream: &mut ReaderStreamRef,
        buffer: &mut Option<ArrayFuture>,
    ) -> VortexResult<Option<ArrayFuture>> {
        if let Some(buffered) = buffer.take() {
            return Ok(Some(buffered));
        }
        stream.next_chunk()
    }
}

/// Skip `n` rows from an optional stream, consuming from the buffer first.
fn skip_child(stream: Option<&mut ReaderStreamRef>, buffer: &mut Option<ArrayFuture>, n: usize) {
    let mut remaining = n;
    if let Some(buf) = buffer.take() {
        if remaining < buf.len() {
            *buffer = Some(buf.slice(remaining..buf.len()));
            return;
        }
        remaining -= buf.len();
    }
    if remaining > 0
        && let Some(stream) = stream
    {
        stream.skip(remaining);
    }
}

impl ReaderStream for StructReaderStream {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn skip(&mut self, n: usize) {
        // Skip n rows from each child (validity + fields), consuming from buffers first.
        skip_child(self.validity.as_mut(), &mut self.validity_buffer, n);
        for (field_stream, field_buf) in self.fields.iter_mut().zip(self.field_buffers.iter_mut()) {
            skip_child(Some(field_stream), field_buf, n);
        }
    }

    fn next_chunk(&mut self) -> VortexResult<Option<ArrayFuture>> {
        // Collect an ArrayFuture for each child (validity + fields).
        let mut all_futures: Vec<ArrayFuture> = Vec::with_capacity(1 + self.fields.len());

        // Validity
        let has_validity = self.validity.is_some();
        if let Some(ref mut validity_stream) = self.validity {
            let Some(validity_future) =
                Self::next_for_child(validity_stream, &mut self.validity_buffer)?
            else {
                return Ok(None);
            };
            all_futures.push(validity_future);
        }

        // Fields
        for (field_stream, field_buf) in self.fields.iter_mut().zip(self.field_buffers.iter_mut()) {
            let Some(field_future) = Self::next_for_child(field_stream, field_buf)? else {
                return Ok(None);
            };
            all_futures.push(field_future);
        }

        if all_futures.is_empty() {
            return Ok(None);
        }

        // Find the minimum length.
        let min_len = all_futures.iter().map(|f| f.len()).min().unwrap_or(0);
        if min_len == 0 {
            return Ok(None);
        }

        // For children with len > min_len, buffer the remainder and slice.
        let mut chunk_futures: Vec<ArrayFuture> = Vec::with_capacity(all_futures.len());

        for (buf_idx, future) in all_futures.into_iter().enumerate() {
            if future.len() > min_len {
                // Buffer the remainder.
                let remainder = future.slice(min_len..future.len());
                let chunk = future.slice(0..min_len);

                if buf_idx == 0 && has_validity {
                    self.validity_buffer = Some(remainder);
                } else {
                    let field_idx = if has_validity { buf_idx - 1 } else { buf_idx };
                    self.field_buffers[field_idx] = Some(remainder);
                }

                chunk_futures.push(chunk);
            } else {
                chunk_futures.push(future);
            }
        }

        let struct_fields = self
            .dtype
            .as_struct_fields_opt()
            .ok_or_else(|| vortex_error::vortex_err!("Expected struct dtype"))?
            .clone();
        let nullability = self.dtype.nullability();
        let estimated_bytes: usize = chunk_futures.iter().map(|f| f.estimated_bytes()).sum();

        Ok(Some(ArrayFuture::new(
            min_len,
            estimated_bytes,
            async move {
                // Split off validity future from field futures.
                let arrays = try_join_all(chunk_futures).await?;

                let (validity, fields) = if has_validity {
                    let validity_array = arrays[0].clone();
                    let fields = arrays[1..].to_vec();
                    (Validity::Array(validity_array), fields)
                } else {
                    (nullability.into(), arrays)
                };

                Ok(
                    StructArray::try_new_with_dtype(fields, struct_fields, min_len, validity)?
                        .into_array(),
                )
            },
        )))
    }
}
