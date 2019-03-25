function [images, labels] = load_mnist_data(imgFile, labelFile)

%% Read the Images (X)
% returns a 28x28x[number of MNIST images] matrix containing the raw MNIST images

% open the file
fp = fopen(imgFile, 'rb');

% smoke test of a file
assert(fp ~= -1, ['Could not open ', imgFile, '']);
magic  = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', imgFile, '']);

% start reading
numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
numRows   = fread(fp, 1, 'int32', 0, 'ieee-be');
numCols   = fread(fp, 1, 'int32', 0, 'ieee-be');
images    = fread(fp, inf, 'unsigned char');
images    = reshape(images, numCols, numRows, numImages);
images    = permute(images,[2 1 3]);

% close file
fclose(fp);

% Reshape to #pixels x #examples
images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
% Convert to double and rescale to [0,1]
images = double(images) / 255;

%% Read the Labels (Y)
% returns a [number of MNIST images]x1 matrix containing the labels for the MNIST images

% open the file
fp = fopen(labelFile, 'rb');

% perform sanity check of a file
assert(fp ~= -1, ['Could not open ', labelFile, '']);
magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2049, ['Bad magic number in ', labelFile, '']);

% start reading
numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');
labels    = fread(fp, inf, 'unsigned char');

% another smoke test
assert(size(labels,1) == numLabels, 'Mismatch in label count');

% close the file
fclose(fp);

end
