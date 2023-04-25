use crate::math::*;

#[derive(Clone, Copy)]
pub struct Projection {
    scale: DVec2,
    translate: DVec2,
}

impl Projection {

    pub const IDENTITY: Self = Self::new(DVec2::ONE, DVec2::ZERO);

    pub const fn new(scale: DVec2, translate: DVec2) -> Self {
        Self { scale, translate }
    }

    pub fn project(&self, coord: DVec2) -> DVec2 {
        self.scale * (coord + self.translate)
    }
    
    pub fn unproject(&self, coord: DVec2) -> DVec2 {
        coord / self.scale - self.translate
    }
    
    pub fn project_vector(&self, vector: DVec2) -> DVec2 {
        self.scale * vector
    }
    
    pub fn unproject_vector(&self, vector: DVec2) -> DVec2 {
        vector / self.scale
    }
    
    pub fn project_x(&self, x: f64) -> f64 {
        self.scale.x * (x + self.translate.x)
    }
    
    pub fn project_y(&self, y: f64) -> f64 {
        self.scale.y * (y + self.translate.y)
    }
    
    pub fn unproject_x(&self, x: f64) -> f64 {
        x / self.scale.x - self.translate.x
    }
    
    pub fn unproject_y(&self, y: f64) -> f64 {
        y / self.scale.y - self.translate.y
    }
}