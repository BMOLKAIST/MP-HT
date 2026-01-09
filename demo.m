%% Fig.3 Demo: MP-HT segmentation + lumen/epithelium separation + dry-mass quantification
% -------------------------------------------------------------------------
%
% Required files/functions:
%   - dataset_fig3.mat (must contain RI, NAc, NAo, ResolutionX, ResolutionZ,
%                       RImedium, wavelength)
%   - MP_HT.m
%   - showOrthoslices.m (for visualization)
%
% Toolboxes:
%   - Image Processing Toolbox
%   - Parallel Computing Toolbox (optional; for gpuArray)
% -------------------------------------------------------------------------

%% 1) Data loading + ROI cropping
load('dataset_fig3.mat');

% Crop ranges (row/y, col/x)
crop_range = cell(1,2);
crop_range{1} = 419:1387;   % y-range (rows)
crop_range{2} = 916:1848;   % x-range (cols)

RI_cropped = RI(crop_range{1}, crop_range{2}, :);

% Axial-to-lateral voxel-size ratio (dz/dx)
scalefactor = ResolutionZ / ResolutionX;


%% 2) Whole-organoid segmentation (MP-HT -> preprocess -> Otsu)
% MP-HT filtering is applied to the full RI volume, then cropped.
filtered = MP_HT(RI, NAc, NAo, ResolutionX, ResolutionZ, RImedium, wavelength);
filtered = filtered(crop_range{1}, crop_range{2}, :);

% Preprocessing (as in the original script):
%   1) downsample XY by 1/scalefactor (approx. isotropic)
%   2) normalize to [0,1] with mat2gray
%   3) 3D median filter (5x5x5)
%   4) upsample back to original size (linear)
%   5) subtract per-(x,y) minimum along z
filtered_iso = imresize3(filtered, Scale = [1/scalefactor 1/scalefactor 1]);
filtered_iso = mat2gray(filtered_iso);
filtered_iso = medfilt3(filtered_iso, [5 5 5]);

enhanced_img = imresize3(filtered_iso, size(filtered), 'method', 'linear');
enhanced_img = enhanced_img - min(enhanced_img, [], 3);

% Initial organoid mask via global Otsu threshold
mask = enhanced_img > graythresh(enhanced_img);


%% 2b) Phantom-based boundary thinning (Appendix-style correction)
% Rationale:
%   MP-HT + Otsu tends to produce a slightly thick boundary. We estimate the
%   boundary bias via a synthetic RI phantom and choose an erosion radius
%   that best matches the original mask (IoU maximization).

% Mean RI inside mask and background RI (medium)
meanRI    = mean(single(RI_cropped(mask)));
meanRI_bg = RImedium;

% Local texture estimate: local std in a 5x5x1 window (xy only)
win = ones(5,5,1);
N = numel(win);
mean_local    = convn(single(RI_cropped),   win/N, 'same');
mean_sq_local = convn(single(RI_cropped).^2, win/N, 'same');
var_local     = mean_sq_local - mean_local.^2;
localSTD      = sqrt(max(var_local, 0));

stdRI    = median(localSTD(mask));
stdRI_bg = median(localSTD(~mask));

clear mean_local mean_sq_local var_local localSTD

% Synthetic RI phantom (do NOT touch RNG here; keep identical behavior)
RI_phantom = (meanRI + stdRI   * randn(size(RI_cropped))) .* mask + ...
             (meanRI_bg + stdRI_bg * randn(size(RI_cropped))) .* (~mask);

% Run the same MP-HT + preprocess + Otsu pipeline on the phantom
filtered_phantom = MP_HT(RI_phantom, NAc, NAo, ResolutionX, ResolutionZ, RImedium, wavelength);

filtered_phantom_iso = imresize3(filtered_phantom, Scale = [1/scalefactor 1/scalefactor 1]);
filtered_phantom_iso = mat2gray(filtered_phantom_iso);
filtered_phantom_iso = medfilt3(filtered_phantom_iso, [5 5 5]);

enhanced_img_phantom = imresize3(filtered_phantom_iso, size(filtered_phantom), 'method', 'linear');
enhanced_img_phantom = enhanced_img_phantom - min(enhanced_img_phantom, [], 3);

mask_phantom = enhanced_img_phantom > graythresh(enhanced_img_phantom);

% Search erosion radius (ellipsoid) that maximizes IoU(mask, eroded(mask_phantom))
r_list = 1:8;
IoU = [];
for r = r_list
    rz = round(r/scalefactor);
    [x,y,z] = meshgrid(-r:r, -r:r, -rz:rz);
    ellipsoidMask = (x/r).^2 + (y/r).^2 + (z/rz).^2 <= 1;
    seEllipsoid = strel('arbitrary', ellipsoidMask);

    maski = imerode(mask_phantom, seEllipsoid);
    IoU(end+1) = sum(mask & maski, 'all') / sum(mask | maski, 'all');
end

[~, m_ind] = max(IoU);
r = r_list(m_ind)  % keep console output identical

rz = round(r/scalefactor);
[x,y,z] = meshgrid(-r:r, -r:r, -rz:rz);
ellipsoidMask = (x/r).^2 + (y/r).^2 + (z/rz).^2 <= 1;
seEllipsoid = strel('arbitrary', ellipsoidMask);

% Apply the chosen thinning to the real mask
mask_orig = mask;
mask = imerode(mask, seEllipsoid);


%% 2c) Quick visual check (single z-slice)
zc = 79;
figure;

subplot(2,5,1)
imagesc(flip(single(RI_cropped(:,:,zc)),1)); clim([1.31 1.38]); axis image off

subplot(2,5,2)
imagesc(flip(filtered(:,:,zc),1)); clim([-8 -4]); axis image off

subplot(2,5,3)
imagesc(flip(enhanced_img(:,:,zc),1)); axis image off

subplot(2,5,4)
imagesc(flip(mask_orig(:,:,zc),1)); axis image off

subplot(2,5,5)
imagesc(flip(mask(:,:,zc),1)); axis image off

subplot(2,5,6)
imagesc(flip(single(RI_phantom(:,:,zc)),1)); clim([1.31 1.38]); axis image off

subplot(2,5,7)
imagesc(flip(filtered_phantom(:,:,zc),1)); clim([-8 -4]); axis image off

subplot(2,5,8)
imagesc(flip(enhanced_img_phantom(:,:,zc),1)); axis image off

subplot(2,5,9)
imagesc(flip(mask_phantom(:,:,zc),1)); axis image off

colormap gray


%% 3) Lumen and epithelium separation 
%% 3a) Initial voxel classification.
% Distance transform is computed in (approximately) isotropic space,
% then resized back to the original grid.
mask_dist = imresize3(bwdist(imresize3(mask, Scale = [1 1 scalefactor])), size(mask));

% Seed detection via extended maxima + border clearing
th = 8;
seed = imextendedmax(mask_dist, th);
seed = imclearborder(seed, 26);

stats = regionprops3(seed, "Volume", "Centroid", "VoxelIdxList");

% Sort connected components by descending volume
[~, order] = sort(stats.Volume, 'descend');
stats = stats(order, :);

% Keep up to N seed components (after a simple bounding-box sanity check)
N = 3;
props = regionprops3(mask, "BoundingBox");

% Build a union-of-bounding-box mask (same behavior as the original script)
sz = size(mask);
bboxMask = false(sz);
for i = 1:height(props)
    bb = props.BoundingBox(i,:);   % [x, y, z, width, height, depth]

    % Convert to integer index ranges (BoundingBox has 0.5 offset semantics)
    x1 = floor(bb(1)) + 1;
    y1 = floor(bb(2)) + 1;
    z1 = floor(bb(3)) + 1;

    x2 = ceil(bb(1) + bb(4));
    y2 = ceil(bb(2) + bb(5));
    z2 = ceil(bb(3) + bb(6));

    bboxMask(max(1,x1):min(x2,end), max(y1,1):min(y2,end), max(z1,1):min(z2,end)) = true;
end

cents = [];
compMask_total = false(size(mask));
for k = 1:size(stats.Centroid,1)
    compMask = false(size(mask));
    compMask(stats.VoxelIdxList{k}) = true;

    % Skip components that extend outside the bounding-box union
    if any(~bboxMask & compMask, 'all')
        continue;
    end

    cents(end+1,:) = stats.Centroid(k,:);
    compMask_total = compMask_total | compMask;

    if size(cents,1) == N
        break;
    end
end

% Convert centroids to integer voxel indices (row=y, col=x, slice=z)
if ~isempty(cents)
    y = round(cents(:,2));
    x = round(cents(:,1));
    z = round(cents(:,3));

    sz = size(seed);
    y = max(1, min(sz(1), y));
    x = max(1, min(sz(2), x));
    z = max(1, min(sz(3), z));

    centroid_idx = [y, x, z];
else
    centroid_idx = [];
end

bg = false(size(mask_dist));
lumen = false(size(mask_dist));

for center = 1:size(centroid_idx,1)

    % Keep only the largest connected component of the organoid mask
    CC = bwconncomp(mask, 26);
    sizes = cellfun(@numel, CC.PixelIdxList);
    [~, idxMax] = max(sizes);

    maskc = false(size(mask));
    if ~isempty(CC.PixelIdxList)
        maskc(CC.PixelIdxList{idxMax}) = true;
    end

    % Coordinates of organoid-shell voxels
    [y, x, z] = ind2sub(size(mask), find(maskc));

    % Center voxel
    xc = centroid_idx(center,2);
    yc = centroid_idx(center,1);
    zc = centroid_idx(center,3);

    % Centered coordinates (for surface sampling)
    Xc = x - xc;
    Yc = y - yc;
    Zc = z - zc;

    % Full grid (used later to classify every voxel)
    [X, Y, Z] = meshgrid(1:size(mask,2), 1:size(mask,1), 1:size(mask,3));
    X = X - xc; Y = Y - yc; Z = Z - zc;

    % Convert shell voxels to spherical coordinates
    [azimuth, elevation, r] = cart2sph(Xc, Yc, scalefactor * Zc);

    % Normalize angles:
    %   azimuth in [0, 2pi)
    %   elevation shifted to [0, pi]
    azimuth   = mod(azimuth, 2*pi);
    elevation = elevation + pi/2;

    % Angular bins (50 x 50)
    numAziBins = 50;
    numEleBins = 50;
    aziEdges = linspace(0, 2*pi, numAziBins+1);
    eleEdges = linspace(0, pi,   numEleBins+1);

    rmin = nan(numEleBins, numAziBins);
    rmax = nan(numEleBins, numAziBins);

    % Compute rmin/rmax per (elevation, azimuth) bin
    for aziIdx = 1:numAziBins
        fprintf(' %d %% Completed\n', aziIdx/numAziBins*100)
        for eleIdx = 1:numEleBins
            inBin = azimuth >= aziEdges(aziIdx) & azimuth < aziEdges(aziIdx+1) & ...
                    elevation >= eleEdges(eleIdx) & elevation < eleEdges(eleIdx+1);
            if any(inBin)
                rmin(eleIdx, aziIdx) = min(r(inBin));
                rmax(eleIdx, aziIdx) = max(r(inBin));
            end
        end
    end

    % Padding to avoid boundary artifacts in interpolation
    padding_size = 10;

    % Azimuth padding (periodic wrap)
    rmin_pad = [rmin(:, end-padding_size+1:end), rmin, rmin(:, 1:padding_size)];
    rmax_pad = [rmax(:, end-padding_size+1:end), rmax, rmax(:, 1:padding_size)];

    % Elevation padding (mirror)
    rmin_pad = [flipud(rmin_pad(1:padding_size,:)); rmin_pad; flipud(rmin_pad(end-padding_size+1:end,:))];
    rmax_pad = [flipud(rmax_pad(1:padding_size,:)); rmax_pad; flipud(rmax_pad(end-padding_size+1:end,:))];

    [Xpad, Ypad] = meshgrid(1-padding_size:numAziBins+padding_size, ...
                            1-padding_size:numEleBins+padding_size);

    % Interpolate missing bins (NaNs)
    valid_min = ~isnan(rmin_pad);
    valid_max = ~isnan(rmax_pad);

    rminInterp_pad = rmin_pad;
    rmaxInterp_pad = rmax_pad;

    rminInterp_pad(~valid_min) = griddata(Xpad(valid_min), Ypad(valid_min), rmin_pad(valid_min), ...
        Xpad(~valid_min), Ypad(~valid_min), 'linear');

    rmaxInterp_pad(~valid_max) = griddata(Xpad(valid_max), Ypad(valid_max), rmax_pad(valid_max), ...
        Xpad(~valid_max), Ypad(~valid_max), 'linear');

    % Remaining NaNs -> nearest fill
    rminInterp_pad = fillmissing(rminInterp_pad, 'nearest');
    rmaxInterp_pad = fillmissing(rmaxInterp_pad, 'nearest');

    % Median filtering + remove padding
    rmaxInterp = medfilt2(rmaxInterp_pad, [3,3], "symmetric");
    rmaxInterp = rmaxInterp(padding_size+1:end-padding_size, padding_size+1:end-padding_size);

    rminInterp = medfilt2(rminInterp_pad, [7,7], "symmetric");
    rminInterp = min( ...
        rminInterp_pad(padding_size+1:end-padding_size, padding_size+1:end-padding_size), ...
        rminInterp(padding_size+1:end-padding_size, padding_size+1:end-padding_size));

    % Bin centers
    azVec = (aziEdges(1:end-1) + aziEdges(2:end)) / 2;
    elVec = (eleEdges(1:end-1) + eleEdges(2:end)) / 2;

    Rmax = rmaxInterp;
    Rmin = rminInterp;

    % Build periodic interpolants over (el, az)
    azVec2 = [azVec, azVec(1) + 2*pi];
    Rmax   = [Rmax, Rmax(:,1)];
    Rmin   = [Rmin, Rmin(:,1)];

    Fmax = griddedInterpolant({elVec, azVec2}, Rmax, 'linear', 'nearest');
    Fmin = griddedInterpolant({elVec, azVec2}, Rmin, 'linear', 'nearest');

    % Convert each voxel in the 3D grid to spherical coordinates
    [azV, elV, rV] = cart2sph(X, Y, scalefactor * Z);
    azV = mod(azV, 2*pi);
    elV = elV + pi/2;

    % Surface radii at each voxel's (el, az)
    rSurfmax = Fmax(elV, azV);
    rSurfmin = Fmin(elV, azV);

    % Classify voxels as background / lumen relative to the shell
    bg    = bg    | (rV > rSurfmax);
    lumen = lumen | (rV < rSurfmin);

end

% Keep lumen/background inside the organoid (exclude shell voxels)
bg    = bg    & (~mask) & ~lumen;
lumen = lumen & (~mask);


%% 3b) Dual-sided iterative reconstruction (constrained dilation)
se = strel(ones(3,3,3));

bg_recon    = bg;
lumen_recon = lumen;
prev = 0;
max_iter = 1000;

for k = 1:max_iter
    se = zeros(3,3,3);
    se(:,:,2) = 1;

    % Keep identical behavior (currently always isotropic 3-slice dilation)
    if 1  % mod(k, round(scalefactor))==1
        se(:,:,1:3) = 1;
    end
    se = strel(se);

    lumen_recon = imdilate(lumen_recon, se);
    boundary = lumen_recon & (~mask) & bg_recon;
    lumen_recon = (lumen_recon & (~mask) & ~bg_recon);

    bg_recon = imdilate(bg_recon, se);
    boundary = boundary | (bg_recon & (~mask) & lumen_recon);
    bg_recon = (bg_recon & (~mask) & ~lumen_recon);

    if sum(lumen_recon, 'all') <= prev
        break
    end
    prev = sum(lumen_recon, 'all');
end

% Final epithelium mask includes the original shell + boundary voxels
epithelium = mask | boundary;

% Final lumen is the reconstructed lumen excluding the boundary
lumen = lumen_recon & ~boundary;


%% 3c) Review detected lumens
% Lumen: keep top 1~2 components
CC = bwconncomp(lumen);
numPixels = cellfun(@numel, CC.PixelIdxList);
[~, sortIdx] = sort(numPixels, 'descend');

labeled = labelmatrix(CC);
lumen_clean = labeled == sortIdx(1);
if length(sortIdx) > 1
    lumen_clean = lumen_clean | (labeled == sortIdx(2));
end

% Epithelium: keep the single largest component
CC = bwconncomp(epithelium);
numPixels = cellfun(@numel, CC.PixelIdxList);
[~, sortIdx] = sort(numPixels, 'descend');

labeled = labelmatrix(CC);
layer_clean = labeled == sortIdx(1);


%% 3d) Visualization (orthoslices)
xc = 452;
yc = 640;
zc = 79;

% RI tomogram
showOrthoslices(RI_cropped, xc, yc, zc, scalefactor, [1.31 1.38]);

% MP-HT
showOrthoslices(filtered, xc, yc, zc, scalefactor, [-8 -4]);

% Build a simple colored RI overlay 
RIc = single(RI_cropped);
RIc = (RIc - 1.31) / (1.38 - 1.31);
RIc = min(max(RIc, 0), 1);

lumen_color = [1 0.5 0.5; 0 0 0];
lumen_color = interp1(1:size(lumen_color,1), lumen_color, linspace(1, size(lumen_color,1), 100));

RIc = layer_clean .* RIc + lumen_clean .* ( ...
    reshape(lumen_color(1,:) - lumen_color(end,:), 1,1,1,3) .* RIc + reshape(lumen_color(end,:), 1,1,1,3));

% Segmentation results (binary masks rendered as RGB)
showOrthoslices(layer_clean .* reshape([0.7 0.7 0.7],1,1,1,3) + lumen_clean .* reshape(lumen_color(1,:),1,1,1,3), ...
    xc, yc, zc, scalefactor);

% Segmented RI overlay
showOrthoslices(RIc, xc, yc, zc, scalefactor);


%% 4) Dry mass quantification
% Spatial frequency support (Support) from the system parameters
Lx = size(RI_cropped,2) * ResolutionX;
Ly = size(RI_cropped,1) * ResolutionX;
Lz = size(RI_cropped,3) * ResolutionZ;

dux = 1 / Lx;
duy = 1 / Ly;
duz = 1 / Lz;

ux = (1:size(RI_cropped,2)) - ceil((size(RI_cropped,2)+1)/2);
uy = (1:size(RI_cropped,1)) - ceil((size(RI_cropped,1)+1)/2);
uz = (1:size(RI_cropped,3)) - ceil((size(RI_cropped,3)+1)/2);

ux = dux * reshape(ux, 1, [], 1);
uy = duy * reshape(uy, [], 1, 1);
uz = duz * reshape(uz, 1, 1, []);

Ewaldc = round((sqrt((RImedium/wavelength).^2 - ux.^2 - uy.^2) - (RImedium/wavelength) - uz) / duz) == 0;
Ewaldc = Ewaldc .* ((ux.^2 + uy.^2) < (NAc/wavelength)^2);

Ewaldo = round((sqrt((RImedium/wavelength).^2 - ux.^2 - uy.^2) - (RImedium/wavelength) - uz) / duz) == 0;
Ewaldo = Ewaldo .* ((ux.^2 + uy.^2) < (NAo/wavelength)^2);

Ewaldc = fftn(Ewaldc);
Ewaldo = fftn(Ewaldo);
Support = abs(ifftn(Ewaldc .* conj(Ewaldo)));
Support = Support > 0.5;   % spatial frequency support mask

% Fit two basis spectra (background + RI offset for organoid) to the measured spectrum
RIf = gather(fftn(gpuArray(RI_cropped)));

RI_bg  = ones(size(RIf), 'single');
RI_org = layer_clean;

RI_bgf  = gather(fftn(gpuArray(RI_bg)));
RI_orgf = gather(fftn(gpuArray(single(RI_org))));

A = RI_bgf  .* Support;
B = RI_orgf .* Support;
C = RIf     .* Support;

X = [A(:) B(:)];
y = C(:);

abc = double(X) \ double(y);

n_bg   = real(abc(1));
dn_org = real(abc(2));

% Refractive increment (mL/g)
RII = 0.185;

drymass_density = dn_org / RII;
volume = sum(layer_clean, 'all') * ResolutionX^2 * ResolutionZ;
drymass = drymass_density * volume;

fprintf('Dry-mass density: %.3g g/mL   Dry mass: %.2g ng \n', drymass_density, drymass * 1e-3)
