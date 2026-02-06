// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::ops::Range;
use std::sync::Arc;

use vortex_array::expr::Expression;
use vortex_array::expr::stats::Stat;
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::v2::reader::Reader;
use crate::v2::reader::ReaderRef;
use crate::v2::reader::ReaderStreamRef;

/// A reader that wraps a data reader with per-zone statistics for pruning.
///
/// The zone map reader holds aggregate statistics (min, max, null_count) for each zone
/// of `zone_len` rows. During execute, zone statistics can be used to skip entire zones
/// that cannot match a filter expression.
pub struct ZonedReader {
    data: ReaderRef,
    zone_map: ReaderRef,
    zone_len: usize,
    present_stats: Arc<[Stat]>,
}

impl ZonedReader {
    pub fn new(
        data: ReaderRef,
        zone_map: ReaderRef,
        zone_len: usize,
        present_stats: Arc<[Stat]>,
    ) -> Self {
        Self {
            data,
            zone_map,
            zone_len,
            present_stats,
        }
    }
}

impl Reader for ZonedReader {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        self.data.dtype()
    }

    fn row_count(&self) -> u64 {
        self.data.row_count()
    }

    fn apply(&self, expression: &Expression) -> VortexResult<ReaderRef> {
        // Apply the expression to the data reader.
        let new_data = self.data.apply(expression)?;

        // TODO(ngates): transform the expression into zone-map-compatible pruning checks
        //  (e.g., `x > 5` becomes `max >= 5`). For now, we just pass through the data
        //  and keep the zone map as-is.
        Ok(Arc::new(ZonedReader {
            data: new_data,
            zone_map: self.zone_map.clone(),
            zone_len: self.zone_len,
            present_stats: self.present_stats.clone(),
        }))
    }

    fn execute(&self, row_range: Range<u64>) -> VortexResult<ReaderStreamRef> {
        // TODO(ngates): use zone map to produce a pruning mask, then drive the data stream
        //  with zones that can't match the filter skipped.
        self.data.execute(row_range)
    }
}
