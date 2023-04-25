use crate::math::*;
use crate::eqution_solver::*;
use bitflags::bitflags;
// Parameters for iterative search of closest point on a cubic Bezier curve. Increase for higher precision.
const MSDFGEN_CUBIC_SEARCH_STARTS: usize = 4;
const MSDFGEN_CUBIC_SEARCH_STEPS: usize = 4;
// The proportional amount by which a curve's control point will be adjusted to eliminate convergent corners.
const MSDFGEN_DECONVERGENCE_FACTOR: f64 = 0.000001;

bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct EdgeColor: u8 {
        const BLACK = 0;
        const RED = 1;
        const GREEN = 2;
        const YELLOW = 3;
        const BLUE = 4;
        const MAGNETA = 5;
        const CYAN = 6;
        const WHITE = 7;
    }
}

pub struct EdgeSegment {
    color: EdgeColor,
    geometry: Box<dyn SegmentGeomentry>,
}

impl EdgeSegment {
    pub fn split_in_thirds(self) -> [EdgeSegment; 3] {
        let color = self.color;
        let geometry = self.geometry.split_in_thirds();
        self.geometry.split_in_thirds().map(|geometry| EdgeSegment { color, geometry })
    }

    pub fn decoverage(&mut self, param: isize) {
        self.geometry = self.geometry.decoverage(param);

    }
}

impl std::ops::Deref for EdgeSegment {
    type Target = dyn SegmentGeomentry;
    fn deref(&self) -> &Self::Target {
        self.geometry.as_ref()
    }
}
impl std::ops::DerefMut for EdgeSegment {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.geometry.as_mut()
    }
}

pub trait SegmentGeomentry {
    /// Returns the point on the edge specified by the parameter (between 0 and 1).
    fn point(&self, param: f64) -> DVec2;
    /// Returns the direction the edge has at the point specified by the parameter.
    fn direction(&self, param: f64) -> DVec2;
    /// Returns the change of direction (second derivative) at the point specified by the parameter.
    fn direction_change(&self, param: f64) -> DVec2;
    /// Returns the minimum signed distance between origin and the edge.
    fn signed_distance(&self, origin: DVec2, param: &mut f64) -> SignedDistance;
    /// Converts a previously retrieved signed distance from origin to pseudo-distance.
    // virtual void distanceToPseudoDistance(SignedDistance &distance, Point2 origin, double param) const;
    /// Outputs a list of (at most three) intersections (their X coordinates) with an infinite horizontal scanline at y and returns how many there are.
    fn scanline_intersections(&self, y: f64) -> Intersections;
    /// Adjusts the bounding box to fit the edge segment.
    fn bound(&self) -> DRect;

    /// Reverses the edge (swaps its start point and end point).
    fn reverse(&mut self);
    /// Splits the edge segments into thirds which together represent the original edge.
    fn split_in_thirds(&self) -> [Box<dyn SegmentGeomentry>;3];

    fn decoverage(&self, param: isize) -> Box<dyn SegmentGeomentry>;
}


#[derive(Default, Clone, Copy)]
/// An intersection with the scanline.
pub struct Intersection {
    /// X coordinate.
    pub x: f64,
    /// Normalized Y direction of the oriented edge at the point of intersection.
    pub direction: isize,
}

#[derive(Default)]
pub struct Intersections {
    size: u8,
    data: [Intersection; 3]
}

impl Intersections {
    fn push(&mut self, x: f64, direction: isize) {
        self.data[self.size as usize] = Intersection {
            x, direction
        };
        self.size += 1;
    }
}

pub struct IntersectionsIterator(u8, Intersections);

impl Iterator for IntersectionsIterator {
    type Item = Intersection;
    fn next(&mut self) -> Option<Self::Item> {
        if self.0 >= self.1.size {
            return None
        }
        let idx = self.0 as usize;
        self.0 += 1;
        Some(self.1.data[idx])
    }
}

impl IntoIterator for Intersections {
    type Item = Intersection;
    type IntoIter = IntersectionsIterator;
    fn into_iter(self) -> Self::IntoIter {
        IntersectionsIterator(0, self)
    }
}

pub struct LinearSegment([DVec2;2]);

impl SegmentGeomentry for LinearSegment {
    fn point(&self, param: f64) -> DVec2 {
        let p = self.0;
        p[0].lerp(p[1], param)
    }

    fn direction(&self, _: f64) -> DVec2 {
        let p = self.0;
        p[1]-p[0]
    }
    fn direction_change(&self, param: f64) -> DVec2 {
        DVec2::ZERO
    }

    fn signed_distance(&self, origin: DVec2, param: &mut f64) -> SignedDistance {
        let p = self.0;
        let aq = origin-p[0];
        let ab = p[1]-p[0];
        *param = aq.dot(ab) / ab.dot(ab);
        let eq = p[if *param > 0.5 { 1 } else { 0 }] - origin;
        let endpoint_distance = eq.length();
        if *param > 0. && *param < 1. {
            let ortho_distance =  ab.orthonormal(false).dot(aq);
            if ortho_distance.abs() < endpoint_distance {
                return SignedDistance::new(ortho_distance, 0.);
            }
        }
        SignedDistance::new(
            aq.cross(ab).non_zero_sign() * endpoint_distance,
            ab.normalize().dot(eq.normalize()).abs()
        )
    }
    fn scanline_intersections(&self, y: f64) -> Intersections {
        let p = self.0;
        let mut intersections = Intersections::default();
        if (y >= p[0].y && y < p[1].y) || (y >= p[1].y && y < p[0].y) {
            let param = (y-p[0].y)/(p[1].y-p[0].y);
            intersections.push(
                p[0].x.mix(p[1].x, param),
                (p[1].y - p[0].y).sign() as isize
            );
        }
        intersections
    }
    fn bound(&self) -> DRect {
        DRect::from_corners(self.0[0], self.0[1])
    }

    fn reverse(&mut self) {
        let p = &mut self.0;
        (p[0], p[2]) = (p[2], p[0]);
    }

    fn split_in_thirds(&self) -> [Box<dyn SegmentGeomentry>;3] {
        let p = self.0;
        [ 
            Box::new(LinearSegment([p[0], self.point(1./3.)])),
            Box::new(LinearSegment([self.point(1./3.), self.point(2./3.)])),
            Box::new(LinearSegment([self.point(2./3.), p[1]])),
        ]
    }
    fn decoverage(&self, _: isize) -> Box<dyn SegmentGeomentry> {
        Box::new(LinearSegment(self.0.clone()))
    }
}


pub struct QuadraticSegment([DVec2; 3]);
impl SegmentGeomentry for QuadraticSegment {
    fn direction(&self, param: f64) -> DVec2 {
        let p = self.0;
        let tangent = (p[1]-p[0]).lerp(p[2]-p[1], param);
        if tangent.is_zero() {
            p[2]-p[0]
        } else {
            tangent
        }
    }
    fn direction_change(&self, param: f64) -> DVec2 {
        let p = self.0;
        (p[2]-p[1])-(p[1]-p[0])
    }
    fn point(&self, param: f64) -> DVec2 {
        let p = self.0;
        p[0].lerp(p[1], param).lerp(p[1].lerp(p[2], param), param)
    }
    fn signed_distance(&self, origin: DVec2, param: &mut f64) -> SignedDistance {
        let p = self.0;
        let qa = p[0]-origin;
        let ab = p[1]-p[0];
        let br = p[2]-p[1]-ab;
        let a = br.dot(br);
        let b = 3. * ab.dot(br);
        let c = 2. * ab.dot(ab) + qa.dot(br);
        let d = qa.dot(ab);
        let mut ep_dir = self.direction(0.);
        let mut min_distance = ep_dir.cross(qa).non_zero_sign() * qa.length(); // distance from A
        *param = -qa.dot(ep_dir) / ep_dir.dot(ep_dir);
        {
            ep_dir = self.direction(1.);
            let distance = (p[2]-origin).length(); // distance from B
            if distance < min_distance.abs() {
                min_distance = ep_dir.cross(p[2]-origin).non_zero_sign() * distance;
                *param = (origin-p[1]).dot(ep_dir) / ep_dir.dot(ep_dir);
            }
        }
        for v in solve_cubic(a, b, c, d) {
            if v > 0. && v < 1. {
                let qe = qa+2.*v*ab+v*v*br;
                let distance = qe.length();
                if distance <= min_distance.abs() {
                    min_distance = (ab+v*br).cross(qe).non_zero_sign()*distance;
                    *param = v;
                }
            }
        }
    
        if *param >= 0. && *param <= 1. {
            SignedDistance::new(min_distance, 0.)
        } else if *param < 0.5 {
            SignedDistance::new(
                min_distance, 
                self.direction(0.).normalize().dot(qa.normalize()).abs()
            )
        } else {
            SignedDistance::new(
                min_distance,
                self.direction(1.).normalize().dot((p[2]-origin).normalize()).abs()
            )
        }
    }

    fn scanline_intersections(&self, y: f64) -> Intersections {
        let mut total = 0;
        let mut x = [0., 0., 0.,];
        let mut dy = [0, 0, 0,];
        let p = self.0;
        let mut next_dy = if y > p[0].y { 1 } else { -1 };
        if self.0[0].y == y {
            if p[0].y < p[1].y || (p[0].y == p[1].y && p[0].y < p[2].y) {
                dy[total] = 1;
                total += 1;
            } else {
                next_dy = 1;
            }
        }
        {
            let ab = p[1] - p[0];
            let br = p[2] - p[1] - ab;
            for v in solve_quadratic(br.y, 2.*ab.y, p[0].y-y) {
                if v > 0. && v < 1. {
                // if (t[i] >= 0 && t[i] <= 1) {
                    x[total] = p[0].x+2.*v*ab.x+v*v*br.x;
                    // x[total] = p[0].x+2*t[i]*ab.x+t[i]*t[i]*br.x;
                    if next_dy as f64 * (ab.y+v*br.y) >= 0. {
                        dy[total] = next_dy;
                        total += 1;
                        next_dy = -next_dy;
                    }
                }
            }
        }

        if p[2].y == y {
            if next_dy > 0 && total > 0 {
                total -= 1;
                next_dy = -1;
            }
            if (p[2].y < p[1].y || (p[2].y == p[1].y && p[2].y < p[0].y)) && total < 2 {
                x[total] = p[2].x;
                if next_dy < 0 {
                    dy[total] = -1;
                    total += 1;
                    next_dy = 1;
                }
            }
        }

        if next_dy != if y >= p[2].y { 1 } else { -1 } {
            if total > 0 {
                total -= 1;
            } else {
                if (p[2].y-y).abs() < (p[0].y-y).abs() {
                    x[total] = p[2].x;
                }
                dy[total] = next_dy;
                total += 1;
            }
        }

        let mut intersections = Intersections::default();
        for i in 0..total {
            intersections.push(x[i], dy[i] );
        }
        intersections
    }
    fn bound(&self) -> DRect {
        let p = self.0;
        let mut bounds = DRect::from_corners(p[0], p[2]);
        let bot = (p[1]-p[0])-(p[2]-p[1]);
        if bot.x != 0. {
            let param = (p[1].x-p[0].x)/bot.x;
            if param > 0. && param < 1. {
                bounds.extend(self.point(param))
            }
        }
        if bot.y != 0. {
            let param = (p[1].y-p[0].y)/bot.y;
            if param > 0. && param < 1. {
                bounds.extend(self.point(param));
            }
        }
        bounds
    }
    fn reverse(&mut self) {
        let p = &mut self.0;
        (p[0], p[1]) = (p[1], p[0]);
    }

    fn split_in_thirds(&self) -> [Box<dyn SegmentGeomentry>;3] {
        let p = self.0;
        [
            Box::new(QuadraticSegment([p[0], p[0].lerp(p[1], 1./3.), self.point(1./3.)])),
            Box::new(QuadraticSegment([
                self.point(1./3.), 
                p[0].lerp(p[1], 5./9.).lerp(p[1].lerp(p[2], 4./9.), 0.5), 
                self.point(2./3.),
            ])),
            Box::new(QuadraticSegment([self.point(2./3.), p[1].lerp(p[2], 2./3.), p[2]]))
        ]
    }
    // convert to cubic
    fn decoverage(&self, _: isize) -> Box<dyn SegmentGeomentry> {
        let p = self.0;
        Box::new(CubicSegment([p[0], p[0].lerp(p[1], 2./3.), p[1].lerp(p[2], 1./3.), p[2]]))
    }


}

pub struct CubicSegment([DVec2; 4]);
impl SegmentGeomentry for CubicSegment {
    fn direction(&self, param: f64) -> DVec2 {
        let p = self.0;
        let tangent = (p[1]-p[0]).lerp(p[2]-p[1], param).lerp((p[2]-p[1]).lerp(p[3]-p[2], param), param);
        if tangent.is_zero() {
            if param == 0. { return p[2]-p[0]; }
            if param == 1. { return p[3]-p[1]; }
        }
        tangent
    }
    fn direction_change(&self, param: f64) -> DVec2 {
        let p = self.0;
        ((p[2]-p[1])-(p[1]-p[0])).lerp((p[3]-p[2])-(p[2]-p[1]), param)
    }
    fn point(&self, param: f64) -> DVec2 {
        let p = self.0;
        let p12 = p[1].lerp(p[2], param);
        p[0].lerp(p[1], param).lerp(p12, param).lerp(
            p12.lerp(p[2].lerp(p[3], param), param), 
            param
        )
    }
    fn signed_distance(&self, origin: DVec2, param: &mut f64) -> SignedDistance {
        let p = self.0;
        let qa = p[0]-origin;
        let ab = p[1]-p[0];
        let br = p[2]-p[1]-ab;
        let az = (p[3]-p[2])-(p[2]-p[1])-br;
    
        let mut ep_dir = self.direction(0.);
        let mut min_distance = ep_dir.cross(qa).non_zero_sign() * qa.length(); // distance from A
        *param = -qa.dot(ep_dir)/ep_dir.dot(ep_dir);
        {
            ep_dir = self.direction(1.);
            let distance = (p[3]-origin).length(); // distance from B
            if distance < min_distance.abs() {
                min_distance = ep_dir.cross(p[3]-origin).non_zero_sign() * distance;
                *param = (ep_dir-(p[3]-origin)).dot(ep_dir)/ep_dir.dot(ep_dir);
            }
        }
        // Iterative minimum distance search
        for i in 0..MSDFGEN_CUBIC_SEARCH_STARTS {
            let mut t = i as f64 / MSDFGEN_CUBIC_SEARCH_STARTS as f64;
            let mut qe = qa+3.*t*ab+3.*t*t*br+t*t*t*az;
            for _ in 0..MSDFGEN_CUBIC_SEARCH_STEPS {
                // Improve t
                let d1 = 3.*ab+6.*t*br+3.*t*t*az;
                let d2 = 6.*br+6.*t*az;
                t -= qe.dot(d1)/(d1.dot(d1)+qe.dot(d2));
                if t <= 0. || t >= 1. {
                    break;
                }
                qe = qa+3.*t*ab+3.*t*t*br+t*t*t*az;
                let distance = qe.length();
                if distance < min_distance.abs() {
                    min_distance = d1.cross(qe).non_zero_sign() * distance;
                    *param = t;
                }
            }
        }
    
        if *param >= 0. && *param <= 1. {
            SignedDistance::new(min_distance, 0.)
        } else if *param < 0.5 {
            SignedDistance::new(min_distance, self.direction(0.0).normalize().dot(qa.normalize()).abs())
        } else {
            SignedDistance::new(min_distance, self.direction(1.0).normalize().dot((p[3]-origin).normalize()).abs())
        }
    }
    fn scanline_intersections(&self, y: f64) -> Intersections {
        let mut total = 0;
        let p = self.0;
        let mut next_dy = if y > p[0].y { 1 } else { -1 };
        let mut x = [0., 0., 0.,];
        let mut dy = [0, 0, 0];
        x[total] = p[0].x;
        if p[0].y == y {
            if p[0].y < p[1].y || (p[0].y == p[1].y && (p[0].y < p[2].y || (p[0].y == p[2].y && p[0].y < p[3].y))) {
                dy[total] = 1;
                total += 1;
            } else {
                next_dy = 1;
            }
        }
        {
            let ab = p[1]-p[0];
            let br = p[2]-p[1]-ab;
            let az = (p[3]-p[2])-(p[2]-p[1])-br;
            for v in solve_cubic(az.y, 3.*br.y, 3.*ab.y, p[0].y-y) {
                if v >= 0. && v <= 1. {
                    x[total] = p[0].x+3.*v*ab.x+3.*v*v*br.x+v*v*v*az.x;
                    if next_dy as f64 * (ab.y+2.*v*br.y+v*v*az.y) >= 0. {
                        dy[total] = next_dy;
                        total += 1;
                        next_dy = -next_dy;
                    }
                }
            }
        }
        if p[3].y == y {
            if next_dy > 0 && total > 0 {
                total -= 1;
                next_dy = -1;
            }
            if (p[3].y < p[2].y || (p[3].y == p[2].y && (p[3].y < p[1].y || (p[3].y == p[1].y && p[3].y < p[0].y)))) && total < 3 {
                x[total] = p[3].x;
                if next_dy < 0 {
                    dy[total] = -1;
                    total += 1;
                    next_dy = 1;
                }
            }
        }
        if next_dy != if y >= p[3].y { 1 } else { -1 } {
            if total > 0 {
                total -= 1;
            } else {
                if (p[3].y-y).abs() < (p[0].y-y).abs() {
                    x[total] = p[3].x;
                }
                dy[total] = next_dy;
                total += 1;
            }
        }
        let mut intersections = Intersections::default();
        for i in 0..total {
            intersections.push(x[i], dy[i]);
        }
        intersections
    }

    fn bound(&self) -> DRect {
        let p = self.0;
        let mut bounds = DRect::from_corners(p[0], p[3]);
        let a0 = p[1]-p[0];
        let a1 = 2.*(p[2]-p[1]-a0);
        let a2 = p[3]-3.*p[2]+3.*p[1]-p[0];
        for v in solve_quadratic(a2.x, a1.x, a0.x) {
            if v > 0. && v < 1. {
                bounds.extend(self.point(v))
            }
        }
        for v in solve_quadratic(a2.y, a1.y, a0.y) {
            if v > 0. && v < 1. {
                bounds.extend(self.point(v))
            }
        }
        bounds
    }
    fn reverse(&mut self) {
        let p = &mut self.0;
        (p[0], p[3]) = (p[3], p[0]);
        (p[1], p[2]) = (p[2], p[1]);
            
    }
    fn split_in_thirds(&self) -> [Box<dyn SegmentGeomentry>;3] {
        let p = self.0;
        [
            Box::new(CubicSegment([
                p[0],
                if p[0] == p[1] { p[0] } else { p[0].lerp(p[1], 1./3.) },
                p[0].lerp(p[1], 1./3.).lerp(p[1].lerp(p[2], 1./3.), 1./3.),
                self.point(1./3.)
            ])),
            Box::new(CubicSegment([
                self.point(1./3.),
                p[0].lerp(p[1], 1./3.).lerp(p[1].lerp(p[2], 1./3.), 1./3.).lerp(
                    p[1].lerp(p[2], 1./3.).lerp(p[2].lerp(p[3], 1./3.), 1./3.), 2./3.
                ),
                p[0].lerp(p[1], 2./3.).lerp(p[1].lerp(p[2], 2./3.), 2./3.).lerp(
                    p[1].lerp(p[2], 2./3.).lerp(p[2].lerp(p[3], 2./3.), 2./3.), 1./3.
                ),
                self.point(2./3.)
            ])),
            Box::new(CubicSegment([
                self.point(2./3.),
                p[1].lerp(p[2], 2./3.).lerp(p[2].lerp(p[3], 2./3.), 2./3.),
                if p[2] == p[3]  { p[3] } else { p[2].lerp(p[3], 2./3.) },
                p[3]
            ]))
        ]
    }
    fn decoverage(&self, param: isize) -> Box<dyn SegmentGeomentry> {
        let mut p = self.0.clone();
        let dir = self.direction(param as f64);
        let normal = dir.orthonormal(true);
        let h = (self.direction_change(param as f64)-dir).dot(normal);
        if param == 0 {
            p[1] += MSDFGEN_DECONVERGENCE_FACTOR * (dir + h.sign() * h.abs().sqrt() * normal);
        } else if param == 1 {
            p[2] -= MSDFGEN_DECONVERGENCE_FACTOR * (dir - h.sign() * h.abs().sqrt() * normal);
        }
        Box::new(CubicSegment(p))
    }
}

pub struct SignedDistance {
    distance: f64,
    dot: f64,
}

impl SignedDistance {
    fn new(distance: f64, dot: f64) -> Self {
        Self { distance, dot }
    }
}