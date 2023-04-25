use std::cmp::Ordering;

use crate::math::*;
use crate::contour::*;
use crate::scanline::Scanline;

pub struct Shape {
    pub invert_axis: bool,
    contours: Vec<Contour>,
}

// Threshold of the dot product of adjacent edge directions to be considered convergent.
const MSDFGEN_CORNER_DOT_EPSILON: f64 = 0.000001;

impl Shape {
    /// Normalizes the shape geometry for distance field generation.
    pub fn normalize(&mut self) {
        for contour in self.contours.iter_mut() {
            if contour.edges.len() == 1 {
                let edges = contour.edges.pop().unwrap().split_in_thirds();
                contour.edges.extend(edges);
            } else {
                let mut prev = contour.edges.len() - 1;
                for curr in 0..contour.edges.len() {
                    let prev_dir = contour.edges[prev].direction(1.).normalize();
                    let cur_dir = contour.edges[curr].direction(0.).normalize();
                    if prev_dir.dot(cur_dir) < MSDFGEN_CORNER_DOT_EPSILON-1. {
                        contour.edges[prev].decoverage(1);
                        contour.edges[curr].decoverage(0);
                    }
                    prev = curr;
                }
            }
        }
    }

    /// Returns the bounding box that fit the shape.
    pub fn bound(&self) -> Option<DRect> {
        let mut result: Option<DRect> = None;
        for contour in self.contours.iter() {
            let Some(cb) = contour.bound() else {
                continue;
            };
            if let Some(bound) = result.as_mut() {
                bound.extend_rect(cb);
            } else {
                result = Some(cb);
            }
        }
        result
    }
    /// Adjusts the bounding box to fit the shape border's mitered corners.
    pub fn bound_miters(&self, border: f64, miter_limit: f64, polarity: isize) -> Option<DRect> {
        let mut result: Option<DRect> = None;
        for contour in self.contours.iter() {
            let Some(cb) = contour.bound_miters(border, miter_limit, polarity) else {
                continue;
            };
            if let Some(bound) = result.as_mut() {
                bound.extend_rect(cb);
            } else {
                result = Some(cb);
            }
        }
        result
    }

    /// Performs basic checks to determine if the object represents a valid shape.
    pub fn validate(&self) -> bool {
        for contour in self.contours.iter() {
            if contour.edges.is_empty() {
                continue;
            }
            let mut corner = contour.edges.last().unwrap().point(1.);
            for edge in contour.edges.iter() {
                if edge.point(0.) != corner {
                    return false;
                }
                corner = edge.point(1.);
            }
        }
        return true;
    }

    /// Outputs the scanline that intersects the shape at y.
    pub fn scanline(&self, y: f64) -> Scanline {
        let mut intersections = vec![];
        for contour in self.contours.iter() {
            for edge in contour.edges.iter() {
                intersections.extend(edge.scanline_intersections(y))
            }
        }
        Scanline::new(intersections)
    }

    /// Returns the total number of edge segments
    pub fn edge_count(&self) -> usize {
        self.contours.iter().map(|c| c.edges.len()).sum()
    }

    /// Assumes its contours are unoriented (even-odd fill rule). Attempts to orient them 
    /// to conform to the non-zero winding rule.
    pub fn orient_contours(&mut self) {
        // an irrational number to minimize chance of intersecting a corner or other point of interest
        let ratio = 0.5*(5.0_f64.sqrt() - 1.);
        let mut orientations = vec![];
        orientations.resize(self.contours.len(), 0);
        let mut intersections = vec![];
        for i in 0..self.contours.len() {
            if orientations[i] == 0 && !self.contours[i].edges.is_empty() {
                // Find an Y that crosses the contour
                let y0 = self.contours[i].edges.first().unwrap().point(0.).y;
                let mut y1 = y0;
                for edge in self.contours[i].edges.iter() {
                    if y0 != y1 {
                        break;
                    }
                    y1 = edge.point(1.).y;
                }
                // in case all endpoints are in a horizontal line
                if y1 == y0 {
                    for edge in self.contours[i].edges.iter() {
                        y1 = edge.point(ratio).y;
                        if y1 != y0 {
                            break;
                        }
                    }
                }
                let y = y0.mix(y1, ratio);
                // Scanline through whole shape at Y
                // double x[3];
                // int dy[3];
                for j in 0..self.contours.len() {
                    for edge in self.contours[j].edges.iter() {
                        intersections.extend(
                            edge.scanline_intersections(y).into_iter().map(|i| (j, i))
                        )
                    }
                }
                intersections.sort_by(|(_, a), (_, b)| {
                    let x = (a.x - b.x).sign();
                    if x < 0. {
                        Ordering::Less
                    } else if x > 0. {
                        Ordering::Greater
                    } else {
                        Ordering::Equal
                    }
                });
                // Disqualify multiple intersections
                for j in 1..intersections.len() {
                    if intersections[j].1.x == intersections[j-1].1.x {
                        intersections[j].1.direction = 0;
                        intersections[j-1].1.direction = 0;
                    }
                }
                // Inspect scanline and deduce orientations of intersected contours
                for (j, (countour_idx, intersection)) in intersections.iter().enumerate() {
                    if intersection.direction != 0 {
                        orientations[*countour_idx] += 2 * ((j as isize & 1) ^ (if intersection.direction > 0 { 1 } else { 0 })) - 1;
                    }
                }
                intersections.clear();
            }
        }
        // Reverse contours that have the opposite orientation
        for (i, orientation) in orientations.iter().enumerate() {
            if *orientation < 0 {
                self.contours[i].reverse();
            }
        }
    }
}
