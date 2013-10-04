%%
%%From coast
test_types = zeros(1, 0)
cd DS2/data_students/coast/Test/
files = dir('coast*')
cd ../../../..
[test_coasts, test_type] = generate_features(files, 1)
test_types = horzcat(test_types, test_type)

%%
%%From forest
cd DS2/data_students/forest/Test/
files = dir('forest*')
cd ../../../..
[test_forests, test_type] = generate_features(files, 2)
test_types = horzcat(test_types, test_type)

%%
%%From city
cd DS2/data_students/insidecity/Test/
files = dir('insidecity*')
cd ../../../..
[test_cities, test_type] = generate_features(files, 3)
test_types = horzcat(test_types, test_type)

%%
%%From mountain
cd DS2/data_students/mountain/Test/
files = dir('mountain*')
cd ../../../..
[test_mountains, test_type] = generate_features(files, 4)
test_types = horzcat(test_types, test_type)

%%
%%Combining
test_images = [test_coasts; test_forests; test_cities; test_mountains]
test_types = test_types'


%%
%%Normalization
norm_test_images = normalize(test_images)