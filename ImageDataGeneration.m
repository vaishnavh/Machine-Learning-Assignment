%%
%%From coast
types = zeros(1, 0)
cd DS2\data_students\coast\train\
files = dir('coast*')
cd ../../../..
[coasts, type] = generate_features(files, 1)
types = horzcat(types, type)

%%
%%From forest
cd DS2\data_students\forest\train\
files = dir('forest*')
cd ../../../..
[forests, type] = generate_features(files, 2)
types = horzcat(types, type)

%%
%%From city
cd DS2\data_students\insidecity\train\
files = dir('insidecity*')
cd ../../../..
[cities, type] = generate_features(files, 3)
types = horzcat(types, type)

%%
%%From mountain
cd DS2\data_students\mountain\train\
files = dir('mountain*')
cd ../../../..
[mountains, type] = generate_features(files, 4)
types = horzcat(types, type)

%%
%%Combining
images = [coasts; forests; cities; mountains]
types = types'


%%
%%Normalization
norm_images = normalize(images)