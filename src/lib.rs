mod circle_ransac{
    use std::ops::Mul;

    use image::{open, ImageBuffer, Luma};
    use imageproc::{edges::canny, drawing::draw_hollow_circle};
    use nalgebra::{Matrix3, Vector3};
    use rand::Rng;

    pub struct Circle {
        pub x: f32,
        pub y: f32,
        pub r: f32,
    }
    pub struct EdgeDetectionAlgorithm {
        pub lower_threshold: f32,
        pub upper_thresold: f32,
    }
    pub struct ModelQualityAlgorithm {
        pub threshold: f32,
        pub hard_decision: bool,
        pub epsilon: f32,
    }
    
    pub fn read_image(path: &str) -> ImageBuffer<Luma<u8>, Vec<u8>> {
        return open(path).unwrap().into_luma8();
    }

    pub fn edge_detection(img: &ImageBuffer<Luma<u8>, Vec<u8>>, edge_algorithm: &EdgeDetectionAlgorithm) -> ImageBuffer<Luma<u8>, Vec<u8>> {
        canny(&img, edge_algorithm.lower_threshold, edge_algorithm.upper_thresold)
    }

    fn construct_model(pt1: (f32, f32), pt2: (f32, f32), pt3: (f32, f32)) -> Circle{
        let (x1, y1) = pt1;
        let (x2, y2) = pt2;
        let (x3, y3) = pt3;

        let A = Matrix3::new(
            2. * x1, 2. * y1, 1., 
            2. * x2, 2. * y2, 1., 
            2. * x3, 2. * y3, 1.,
        );

        let b = Vector3::new(
            x1 * x1 + y1 * y1, 
            x2 * x2 + y2 * y2,
            x3 * x3 + y3 * y3
        );

        let inv_A = match A.try_inverse() {
            Some(inv) => inv,
            None => panic!("Matrix singular!"),
        };
        let abc_vec = inv_A.mul(b);
        // r = sqrt(c + a^2 + b^2)
        let r = (abc_vec[2] + abc_vec[0] * abc_vec[0] + abc_vec[1] * abc_vec[1]).sqrt(); 

        return Circle{x: abc_vec[0], y: abc_vec[1], r: r};
    }


    fn calculate_support(circ: &Circle, set: &Vec<(usize, usize)>, algorithm: &ModelQualityAlgorithm) -> f32{
        // Todo - use generics
        let mut support = 0.0;

        for (idx, coords) in set.iter().enumerate() {
            let (x, y) = *coords;
            let diff = (((x as f32 - circ.x).powf(2.) + (y as f32 - circ.y).powf(2.)).sqrt() - circ.r).abs();
            if diff < algorithm.threshold {
                if algorithm.hard_decision {
                    support += 1.;
                } else {
                    support += 1. / (diff + algorithm.epsilon);
                }
            } 
        }
        return support;
    }

    fn get_min_K(pmax: f64, aout: f64, m: usize) -> usize {
        debug_assert!(pmax > 0.0 && pmax < 1.0 && aout > 0.0 && aout < 1.0, "pmax and aout must be probabilities");
        return ((pmax.ln()) / (1. - (1. - aout).powf(m as f64)).ln()).ceil() as usize;
    }

    fn is_set(v: &Vec<usize>) -> bool {
        assert!(v.len() == 3);
        return v[0] != v[1] && v[0] != v[2] && v[1] != v[2];
    }

    pub fn ransac_circle_detection(
        img: &ImageBuffer<Luma<u8>, Vec<u8>>,
        edge_algorithm: &EdgeDetectionAlgorithm,
        support_algorithm: &ModelQualityAlgorithm,
        pmax: f64,
        aout: f64,
        m: usize,
    ) -> Circle 
    {
        let edges = canny(&img, edge_algorithm.lower_threshold, edge_algorithm.upper_thresold);

        // save edge coordinates into vector <-> size unknown start with 10 % of image pixel
        let mut edge_pts: Vec<(usize, usize)> = Vec::with_capacity((0.1 * (edges.height() * edges.width()) as f64).ceil() as usize);
        for x in 0..edges.width() {
            for y in 0..edges.height() {
                let value = edges.get_pixel(x, y); 
                if value.0[0] > 0u8 {
                    edge_pts.push((x as usize, y as usize));
                }
            }
        }
        let num_edge_pts = edge_pts.len();
        
        let mut best_support = 0.0;
        let mut best_circ = Circle{x: 0.0, y:0.0, r: 0.0}; 

        for iteration in 0..get_min_K(pmax, aout, m) {
            // draw random idxs from set
            let mut idxs: Vec<usize>;
            loop {
                idxs = (0..3).map(|_| {
                    // todo read doc how inefficient this is..
                    rand::thread_rng().gen_range(0..num_edge_pts)
                }).collect::<Vec<usize>>();
                if is_set(&idxs) {break;}
            }

            // ToDo how todo this properly
            let pt1 = edge_pts[idxs[0]];
            let pt1 = (pt1.0 as f32, pt1.1 as f32);
            let pt2 = edge_pts[idxs[1]];
            let pt2 = (pt2.0 as f32, pt2.1 as f32);
            let pt3 = edge_pts[idxs[2]];
            let pt3 = (pt3.0 as f32, pt3.1 as f32);

            let circ = construct_model(pt1, pt2, pt3);
            let support = calculate_support(&circ, &edge_pts, &support_algorithm);
            if support > best_support {
                best_support = support;
                best_circ = circ;
                println!("New best model in iteration {} with support {}", iteration, support);
            }
        }
        
        return best_circ; 
    }

    pub fn draw_circ(img: &ImageBuffer<Luma<u8>, Vec<u8>>, circ: &Circle, path: &str) {
        let ret = draw_hollow_circle(img, (circ.x.round() as i32, circ.y.round() as i32), circ.r.round() as i32, Luma([255]));
        ret.save(path).expect("Error saving file");
    }
    
    #[cfg(test)]
    mod tests {
        use approx::{relative_eq};
        use crate::circle_ransac::{Circle, ModelQualityAlgorithm, EdgeDetectionAlgorithm};
        use super::*;

        static INPUT_PATH: &'static str = "res/image1.png";

        #[test]
        fn read_image_test() {
            let gray = read_image(INPUT_PATH);
            let (h, w) = gray.dimensions();
        }

        #[test]
        fn edge_detection_imwrite_test() {
            let gray = read_image(INPUT_PATH);
            let alg = EdgeDetectionAlgorithm{lower_threshold: 30., upper_thresold: 80.};
            let edges = edge_detection(&gray, &alg);
            edges.save("/tmp/edges.png").expect("Imwrite failed");
        }

        #[test]
        fn construct_model_test() {
            let pt1 = (5. as f32, 7. as f32);
            let pt2 = (1. as f32, 3. as f32);
            let pt3 = (9. as f32, 2. as f32);

            let circ = construct_model(pt1, pt2, pt3);
            
            assert!(relative_eq!(circ.x, 5.055, epsilon=0.01));
            assert!(relative_eq!(circ.y, 2.944, epsilon=0.01));
            assert!(relative_eq!(circ.r, 4.056, epsilon=0.01));
        }

        #[test]
        fn calculate_support_test() {
            // test single points for their support 
            let circ = Circle{x: 5.055, y: 2.944, r: 4.056};
            let algorithm = ModelQualityAlgorithm{threshold: 0.2, hard_decision: true, epsilon: 0.0};
            
            let input: Vec<(usize, usize)> = vec![(5, 7)];
            let support = calculate_support(&circ, &input, &algorithm);
            assert!(relative_eq!(support, 1.0, epsilon=0.01));

            let input: Vec<(usize, usize)> = vec![(1, 3)];
            let support = calculate_support(&circ, &input, &algorithm);
            assert!(relative_eq!(support, 1.0, epsilon=0.01));

            let input: Vec<(usize, usize)> = vec![(5, 5)];
            let support = calculate_support(&circ, &input, &algorithm);
            assert!(relative_eq!(support, 0.0, epsilon=0.01));

            let algorithm = ModelQualityAlgorithm{threshold: 0.2, hard_decision: false, epsilon: 1.0};

            let input: Vec<(usize, usize)> = vec![(5, 7)];
            let support = calculate_support(&circ, &input, &algorithm);
            assert!(support > 0.0);

            let input: Vec<(usize, usize)> = vec![(1, 3)];
            let support = calculate_support(&circ, &input, &algorithm);
            assert!(support > 0.0);

            let input: Vec<(usize, usize)> = vec![(5, 5)];
            let support = calculate_support(&circ, &input, &algorithm);
            assert!(relative_eq!(support, 0.0, epsilon=0.01));
        }

        #[test]
        fn get_min_K_test() {
            let k = get_min_K(0.01, 0.8, 3);
            assert_eq!(k, 574);
        }
        
        #[test]
        fn is_set_test() {
            let input = vec![1, 2, 3];
            assert!(is_set(&input));

            let input = vec![1, 2, 1];
            assert!(!is_set(&input));

            let input = vec![2, 2, 1];
            assert!(!is_set(&input));
        }

        #[test]
        fn circle_ransac_test() {
            let gray = read_image(INPUT_PATH);
            let support_alg = ModelQualityAlgorithm{threshold: 0.2, hard_decision: true, epsilon: 0.0};
            let edge_alg = EdgeDetectionAlgorithm{lower_threshold: 15., upper_thresold: 45.};
            let pmax = 0.01;
            let aout = 0.8;
            let m = 3;
            let circ = ransac_circle_detection(
                &gray, 
                &edge_alg, 
                &support_alg, 
                pmax, 
                aout, 
                m
            );
            println!("Circ: x={} y={} r={}", circ.x, circ.y, circ.r);
            draw_circ(&gray, &circ, "/tmp/result.png");
            
            assert!(relative_eq!(circ.x, 146., epsilon=2.));
            assert!(relative_eq!(circ.y, 118., epsilon=2.));
            assert!(relative_eq!(circ.r, 91., epsilon=2.));
        }
    }
}


