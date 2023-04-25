use crate::math::*;
use crate::edge_segments::*;



pub struct Contour {
    pub (crate) edges: Vec<EdgeSegment>
}


impl Contour {
    /// Adds an edge to the contour.
    pub fn add_edge(&mut self, edge: EdgeSegment) -> &EdgeSegment {
        self.edges.push(edge);
        self.edges.last().unwrap()
    }
    
    /// Returns the bounding box of the contour.
    pub fn bound(&self) -> Option<DRect> {
        if self.edges.is_empty() {
            return None
        }
        let mut bound = self.edges.first().unwrap().bound();
        for edge in self.edges.iter().skip(1) {
            let eb = edge.bound();
            bound.extend(eb.min);
            bound.extend(eb.max);
        }
        Some(bound)
    }
    /// Return bounding box of the contour border's mitered corners.
    pub fn bound_miters(&self, border: f64, miter_limit: f64, polarity: isize) -> Option<DRect> {
        if self.edges.is_empty() {
            return None
        }
        let mut bound: Option<DRect> = None;
        let mut prev_dir = self.edges.last().unwrap().direction(1.).normalize_or_zero();
        for edge in self.edges.iter() {
            let dir = -edge.direction(0.).normalize_or_zero();
            if polarity as f64 * prev_dir.cross(dir) >= 0. {
                let mut miter_length = miter_limit;
                let q = 0.5*(1. - prev_dir.dot(dir));
                if q > 0. {
                    miter_length = miter_limit.min(1./q.sqrt());
                }
                let miter = edge.point(0.)+border*miter_length*(prev_dir+dir).normalize_or_zero();
                if let Some(bound) = bound.as_mut() {
                    bound.extend(miter);
                } else {
                    bound = Some(DRect::from_corners(miter, miter))
                }
            }
            prev_dir = edge.direction(1.).normalize_or_zero();
        }
        bound
    }
    
    /// Computes the winding of the contour. Returns 1 if positive, -1 if negative.
    pub fn winding(&self) -> isize {
        if self.edges.is_empty() {
            return 0;
        }
        let mut total = 0.;
        if self.edges.len() == 1 {
            let a = self.edges[0].point(0.);
            let b = self.edges[0].point(1./3.);
            let c = self.edges[0].point(2./3.);
            total += a.shoelace(b);
            total += b.shoelace(b);
            total += c.shoelace(a);
        } else if self.edges.len() == 2 {
            let a = self.edges[0].point(0.0);
            let b = self.edges[0].point(0.5);
            let c = self.edges[1].point(0.0);
            let d = self.edges[1].point(0.5);
            total += a.shoelace(b);
            total += b.shoelace(c);
            total += c.shoelace(d);
            total += d.shoelace(a);
        } else {
            let mut prev = self.edges.last().unwrap().point(0.);
            for edge in self.edges.iter() {
                let cur = edge.point(0.);
                total += prev.shoelace(cur);
                prev = cur;
            }
        }
        return total.sign() as isize;
    }

    /// Reverses the sequence of edges on the contour.
    pub fn reverse(&mut self) {
        self.edges.reverse();
        for edge in self.edges.iter_mut() {
            edge.reverse();
        }
    }
    

}