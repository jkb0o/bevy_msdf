pub use bevy::math::DVec2;


pub trait Arithmetics {
    /// Returns the weighted average of self and other.
    fn mix(self, other: Self, weight: Self) -> Self;

    fn sign(&self) -> Self;

    fn non_zero_sign(&self) -> Self;
}

impl Arithmetics for f64 {
    fn mix(self, other: Self, weight: Self) -> Self {
        return (1. - weight) * self + weight * other
    }

    /// Returns 1 for positive values, -1 for negative values, and 0 for zero.
    fn sign(&self) -> Self {
        if self == &0. {
            0.
        } else {
            self.signum()
        }
    }
    
    /// Returns 1 for non-negative values and -1 for negative values.
    fn non_zero_sign(&self) -> Self {
        self.signum()
    }
}

pub trait VecArithmetics {
    /// Returns a vector with the same length that is orthogonal to this one.
    fn orthonormal(&self, polarity: bool) -> Self;

    fn cross(&self, rhs: Self) -> f64;

    fn is_zero(&self) -> bool;

    fn shoelace(&self, rhs: Self) -> f64;
}

impl VecArithmetics for DVec2 {
    fn orthonormal(&self, polarity: bool) -> Self {
        let len = self.length();
        if len == 0. {
            if polarity {
                Self::new(0., 1.)
            } else {
                Self::new(0., 1.)
            }
        } else {
            if polarity {
                Self::new(-self.y/len, self.x/len)
            } else {
                Self::new(self.y/len, -self.x/len)
            } 
        }
    }

    fn is_zero(&self) -> bool {
        self.x == 0. && self.y == 0.
    }

    fn cross(&self, rhs: Self) -> f64 {
        self.perp_dot(rhs)
    }

    fn shoelace(&self, rhs: Self) -> f64 {
        (rhs.x-self.x)*(self.y+rhs.y)
    }

}


pub struct DRect {
    pub min: DVec2,
    pub max: DVec2,
}

impl DRect {
    pub fn new(left: f64, right: f64, top: f64, bottom: f64) -> Self {
        Self { 
            min: DVec2::new(left, top),
            max: DVec2::new(right, bottom)
        }
    }
    pub fn from_corners(min: DVec2, max: DVec2) -> Self {
        let (left, right) = if min.x > max.x {
            (max.x, min.x)
        } else {
            (min.x, max.x)
        };
        let (top, bottom) = if min.y > max.y {
            (max.y, min.y)
        } else {
            (min.y, max.y)
        };
        Self::new(left, right, top, bottom)
    }

    pub fn extend(&mut self, point: DVec2) {
        if self.min.x > point.x { self.min.x = point.x };
        if self.max.x < point.x { self.max.x = point.x };
        if self.min.y > point.y { self.min.y = point.y };
        if self.max.y < point.y { self.max.y = point.y };
    }

    pub fn extend_rect(&mut self, rhs: Self) {
        self.extend(rhs.min);
        self.extend(rhs.max);
    }
}