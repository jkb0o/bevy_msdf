use crate::{projection::Projection, shape::Shape, bitmap::BitMap, scanline::FillRule};

pub fn rasterize(output: &mut BitMap<1>, shape: &Shape, projection: Projection, fill_rule: FillRule) {
    for y in 0..output.height() {
        let row = if shape.invert_axis { output.height() - y - 1 } else { y };
        let scanline = shape.scanline(projection.unproject_y(y as f64 + 0.5));
        for x in 0..output.width() {    
            output.get_mut(x, row)[0] = if scanline.filled(projection.unproject_x(x as f64 + 0.5), fill_rule) {
                1.0
            } else {
                0.0
            }
        }
    }
}

// pub fn distance_sign_correction(sdf: &mut BitMap<1> sdf, shape: &Shape, projection: &Projection, fill_rule: FillRule) {
    
//     for y in 0..sdf.height() {
//         let row = if shape.invert_axis { sdf.height() - y - 1 } else { y };
//         let scanline = shape.scanline(projection.unproject_y(y as f64 + 0.5));
//         for x in 0..sdf.width() {    
//             let fill = scanline.filled(projection.unproject_x(x as f64 + 0.5), fill_rule);
//             float &sd = *sdf(x, row);
//             if ((sd > .5f) != fill)
//                 sd = 1.f-sd;
//         }
//     }
// }