// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt;
use std::fmt::Display;
use std::ops::Range;
use std::sync::Arc;

use termtree::Tree;
use vortex_array::ArrayFuture;
use vortex_array::expr::Expression;
use vortex_dtype::DType;
use vortex_error::VortexResult;

pub type ReaderRef = Arc<dyn Reader>;

/// A reader provides an interface for loading data from row-indexed layouts.
///
/// Readers have a concrete row count allowing fixed partitions over a known set of rows. Readers
/// are driven by calling `next_chunk()` which returns an [`ArrayFuture`] with a known length.
pub trait Reader: 'static + Send + Sync {
    /// Downcast the reader to a concrete type.
    fn as_any(&self) -> &dyn Any;

    /// Get the data type of the layout being read.
    fn dtype(&self) -> &DType;

    /// Returns the number of rows in the reader.
    fn row_count(&self) -> u64;

    /// Apply an expression to the reader, returning a new reader that will execute the expression
    /// on top of the current reader.
    fn apply(&self, expression: &Expression) -> VortexResult<ReaderRef>;

    /// Creates a scan over the given row range of the reader.
    fn execute(&self, row_range: Range<u64>) -> VortexResult<ReaderStreamRef>;

    /// Build a tree representation of this reader for display purposes.
    fn display_tree(&self) -> Tree<String>;
}

/// A convenience wrapper for displaying a reader tree.
pub struct DisplayReaderTree<'a>(pub &'a dyn Reader);

impl Display for DisplayReaderTree<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.display_tree())
    }
}

pub type ReaderStreamRef = Box<dyn ReaderStream>;

/// A stream of array chunks from a reader.
///
/// Each call to `next_chunk()` returns an [`ArrayFuture`] whose length is determined by
/// the reader (not the caller). Returns `None` when the stream is exhausted.
pub trait ReaderStream: 'static + Send + Sync {
    /// The data type of the returned data.
    fn dtype(&self) -> &DType;

    /// Skip the next `n` rows of the stream.
    ///
    /// # Panics
    ///
    /// Panics if `n` is greater than the number of rows remaining in the stream.
    fn skip(&mut self, n: usize);

    /// Returns the next chunk of data as an [`ArrayFuture`], or `None` if no more chunks.
    fn next_chunk(&mut self) -> VortexResult<Option<ArrayFuture>>;
}
