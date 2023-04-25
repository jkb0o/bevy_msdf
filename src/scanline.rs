use std::cell::Cell;
use std::cmp::Ordering;

use crate::edge_segments::Intersection;
use crate::math::*;
use crate::contour::*;

/// Fill rule dictates how intersection total is interpreted during rasterization.
#[derive(Clone, Copy)]
pub enum FillRule {
    NonZero,
    EvenOdd,
    Positive,
    Negative,
}

pub struct Scanline {
    intersections: Vec<Intersection>,
    last_index: Cell<usize>,
}

impl Scanline {
    pub fn new(intersections: Vec<Intersection>) -> Scanline {
        let mut scanline = Scanline { intersections, last_index: Cell::new(0) };
        scanline.preprocess();
        scanline
    }

    fn preprocess(&mut self) {
        self.intersections.sort_by(|a, b| {
            let x = (a.x - b.x).sign();
            if x < 0. {
                Ordering::Less
            } else if x > 0. {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        });
        let mut total_direction = 0;
        for intersection in self.intersections.iter_mut() {
            total_direction += intersection.direction;
            intersection.direction = total_direction;
        }
    }

    fn move_to(&self, x: f64) -> isize {
        if self.intersections.is_empty() {
            return -1;
        }
        let mut index = self.last_index.get();
        if x < self.intersections[index].x {
            loop {
                if index == 0 {
                    self.last_index.replace(0);
                    return -1;
                }
                index -= 1;
                if x < self.intersections[index].x {
                    break;
                }
            }
        } else {
            while index < self.intersections.len()-1 && x >= self.intersections[index+1].x {
                index += 1;
            }
        }
        self.last_index.replace(index);
        index as isize
    }

    fn count_intersections(&self, x: f64) -> isize {
        self.move_to(x)+1
    }
    
    fn sum_intersections(&self, x: f64) -> isize {
        let index = self.move_to(x);
        if index >= 0 {
            self.intersections[index as usize].direction
        } else {
            0
        }
    }

    /// Decides whether the scanline is filled at x based on fill rule.
    pub fn filled(&self, x: f64, fill_rule: FillRule) -> bool {
         interpret_fill_rule(self.sum_intersections(x), fill_rule)
    }
}

fn interpret_fill_rule(intersections: isize, fill_rule: FillRule) -> bool {
    use FillRule::*;
    match fill_rule {
        NonZero => intersections != 0,
        EvenOdd => (intersections & 1) != 0,
        Positive => intersections > 0,
        Negative => intersections < 0,
    }
}