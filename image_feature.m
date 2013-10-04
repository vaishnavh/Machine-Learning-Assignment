function [ feature ] = image_feature( file_name )
%image_feature Returns 1 x 96 feature vector of image
%   Detailed explanation goes here
    im = imread(file_name)
    r = im(:,:,1)
    g = im(:,:,2)
    b = im(:,:,3)
    fr = imhist(r, 32)
    fg = imhist(g, 32)
    fb = imhist(b, 32)
    feature = vertcat(fr, fg, fb)
end

