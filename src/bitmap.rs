
pub struct BitMap<const S: usize> {
    width: usize,
    height: usize,
    data: Vec<[f32; S]>
}

impl<const S: usize> BitMap<S> {
    pub fn new(width: usize, height: usize) -> Self {
        let mut data = vec![];
        data.resize(width * height, [0.; S]);
        Self { width, height, data }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn get(&self, x: usize, y: usize) -> &[f32; S] {
        &self.data[self.width * y + x]
    }

    pub fn get_mut(&mut self, x: usize, y: usize) -> &mut [f32; S] {
        &mut self.data[self.width * y + x]
    }

    pub fn write<'a>(&'a mut self) -> BitMapWriteIterator<'a, S> {
        BitMapWriteIterator {
            x: 0, y: 0, w: self.width, iter: self.data.iter_mut()
        }
    }
}

pub struct BitMapWriteIterator<'a, const S: usize> {
    iter: std::slice::IterMut<'a, [f32; S]>,
    x: usize,
    y: usize,
    w: usize,
}

// impl<'a, const S: usize> BitMapWriteIterator<'a, S> {
//     fn get_mut<'b: 'a>(&'b mut self, idx: usize) -> &'a mut [f32; S] {
//         for x in self.bitmap.data.iter_mut()  {
            
//         }
//         &mut self.bitmap.data[idx]
//     }

// }

impl<'a, const S: usize> Iterator for BitMapWriteIterator<'a, S> {
    type Item = (usize, usize, &'a mut [f32; S]);

    fn next(&mut self) -> Option<Self::Item> {
        let x = self.x;
        let y = self.y;
        self.x += 1;
        if self.x > self.w {
            self.x = 0;
            self.y += 1;
        }
        self.iter.next().map(|d| (x, y, d))
    }
}

