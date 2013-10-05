%%
%%Grid search
best_poly = length(types)
poly_c = 0
poly_g = 0
poly_d = 0
poly_cf = 0
for log2c = [-1, 0, 3, 5, 10]
    for log2g = [-1, 0, 3, 5, 10]
        for log2d = [-1, 0, 1, 2, 3]
            for log2cf = [-1, 0, 2, 4]
                error = polynomial_kernel([2^log2c, 2^log2g, 2^log2d, 2^log2cf], CV, norm_images, types);        
                if (error <= best_poly)
                    best_poly = error
                    poly_c = 2^log2c
                    poly_g = 2^log2g
                    poly_d = 2^log2d
                    poly_cf = 2^log2cf
                end
            end
        end
    end
end