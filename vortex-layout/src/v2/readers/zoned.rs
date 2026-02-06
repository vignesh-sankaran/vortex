// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::ops::Range;
use std::sync::Arc;

use vortex_array::expr::Expression;
use vortex_array::expr::stats::Stat;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::v2::reader::Reader;
use crate::v2::reader::ReaderRef;
use crate::v2::reader::ReaderStreamRef;

pub struct ZonedReader {
    data: ReaderRef,
    zone_map: ReaderRef,
    zone_len: usize,
    present_stats: Arc<[Stat]>,
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
        // We need to apply the expression to both the data and the zone map.
        let new_data = self.data.apply(expression)?;
        let new_zone_map = self.zone_map.apply(expression)?;

        // We also need to update the present stats for the new zone map.
        let new_present_stats = self
            .present_stats
            .iter()
            .map(|stat| match stat {
                Stat::Min => Stat::Min,
                Stat::Max => Stat::Max,
                Stat::NullCount => Stat::NullCount,
                _ => vortex_bail!("Unsupported stat for zoned reader: {:?}", stat),
            })
            .collect();

        Ok(Arc::new(ZonedReader {
            data: new_data,
            zone_map: new_zone_map,
            zone_len: self.zone_len,
            present_stats: Arc::new(new_present_stats),
        }))
    }

    fn execute(&self, row_range: Range<u64>) -> VortexResult<ReaderStreamRef> {
        // By default, a zoned reader is just a pass-through.
        self.data.execute(row_range)
    }
}

/// A reader that expands zoned statistics to match the data rows.
/// This repeats each row of the zone map `zone_len` times.
/// TODO(ngates): we could use a RunEndReader + Slice to do this?
struct ZonedExpansionReader {
    zoned: ReaderRef,
    zone_len: usize,
    row_count: u64,
}

impl Reader for ZonedExpansionReader {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        self.zoned.dtype()
    }

    fn row_count(&self) -> u64 {
        self.row_count
    }

    fn apply(&self, expression: &Expression) -> VortexResult<ReaderRef> {
        todo!()
    }

    fn execute(&self, _row_range: Range<u64>) -> VortexResult<ReaderStreamRef> {
        todo!()
    }
}
