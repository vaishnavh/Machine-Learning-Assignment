function [ normalized_data ] = normalize( data )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    normalized_data = (data - repmat(min(data,[],1),size(data,1),1))*spdiags(1./(max(data,[],1)-min(data,[],1))',0,size(data,2),size(data,2))

end

