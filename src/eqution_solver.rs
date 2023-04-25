use std::f64::consts::PI;

pub struct Roots {
    num_solutions: u8,
    solutions: [f64; 3],
}
impl Roots {
    pub fn none() -> Self {
        Self {
            num_solutions: 0,
            solutions: [0., 0., 0.]
        }
    }
    pub fn one(a: f64) -> Self {
        Self {
            num_solutions: 1,
            solutions: [a, 0., 0.]
        }
    }
    pub fn two(a: f64, b: f64) -> Self {
        let (a, b) = if a > b {(b, a)} else {(a, b)};
        Self {
            num_solutions: 2,
            solutions: [a, b, 0.]
        }
    }
    pub fn three(a: f64, b: f64, c: f64) -> Self {
        let (a, b) = if a > b {(b, a)} else {(a, b)};
        let (b, c) = if b > c {(c, b)} else {(b, c)};
        let (a, b) = if a > b {(b, a)} else {(a, b)};
        Self {
            num_solutions: 3,
            solutions: [a, b, c]
        }
    }

    pub fn slice(&self) -> &[f64] {
        &self.solutions[0..self.num_solutions as usize]
    }

    pub fn iter(&self) -> std::slice::Iter<f64> {
        self.solutions[0..self.num_solutions as usize].iter()
    }
}

pub struct RootsIterator(u8, Roots);

impl Iterator for RootsIterator {
    type Item = f64;
    fn next(&mut self) -> Option<Self::Item> {
        if self.0 >= self.1.num_solutions {
            return None
        }
        let idx = self.0 as usize;
        self.0 += 1;
        Some(self.1.solutions[idx])
    }
}

impl IntoIterator for Roots {
    type Item = f64;
    type IntoIter = RootsIterator;
    fn into_iter(self) -> Self::IntoIter {
        RootsIterator(0, self)
    }
}



pub fn solve_quadratic(a: f64, b: f64, c: f64) -> Roots {
    // a == 0 -> linear equation
    if a == 0. || b.abs() > 1e12*a.abs() {
        // a == 0, b == 0 -> no solution
        if b == 0. {
            if c == 0. {
                return Roots::none(); // 0 == 0
            }
            return Roots::none();
        }
        return Roots::one(-c/b);
    }
    let dscr = b*b-4.*a*c;
    if dscr > 0. {
        let dscr = dscr.sqrt();
        return Roots::two(
            (-b+dscr)/(2.*a),
            (-b-dscr)/(2.*a)
        );
    } else if dscr == 0. {
        return Roots::one(-b/(2.*a));
    } else {
        return Roots::none();
    }
}

pub fn solve_cubic_normed(a: f64, b: f64, c: f64) -> Roots {
    let a2 = a*a;
    let q = 1./9.*(a2-3.*b);
    let r = 1./54.*(a*(2.*a2-9.*b)+27.*c);
    let r2 = r*r;
    let q3 = q*q*q;
    let a = a/3.;
    if r2 < q3 {
        let t = (r/q3.sqrt()).clamp(-1., 1.).acos();
        let q = -2.*q.sqrt();
        return Roots::three(
            q*(1./3.*t).cos()-a,
            q*(1./3.*(t+2.*PI)).cos()-a,
            q*(1./3.*(t-2.*PI)).cos()-a,
        )
    } else {
        
        let u = (if r < 0. { 1. } else { -1. })*(r.abs()*(r2-q3).sqrt()).powf(1./3.);
        let v = if u == 0. { 0. } else { q/u };
        let x0 = (u+v)-a;
        if u == v || (u-v).abs() < 1e-12*(u+v).abs() {
            return Roots::two(u, -0.5*(u+v)-a)
        } else {
            return Roots::one(x0)
        }
    }
}

pub fn solve_cubic(a: f64, b: f64, c: f64, d: f64) -> Roots {
    if a != 0. {
        let bn = b/a;
        if bn.abs() < 1e6 { // Above this ratio, the numerical error gets larger than if we treated a as zero
            return solve_cubic_normed(bn, c/a, d/a);
        }
    }
    return solve_quadratic(b, c, d);
}