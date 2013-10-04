function [ features, type ] = generate_features( files, class)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
features = zeros(0,32)
type = []
for file = files'
    feature = image_feature(file.name) 
    type = horzcat(type, class)
    features = vertcat(features, feature')
end



end

