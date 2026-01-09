function RI_filtered_log = MP_HT(RI, NAc, NAo, ResolutionX, ResolutionZ, RImedium, wavelength)
%MP_HT  Multi-Peak Holotomography (MP-HT) transform used in the Fig.3 demo.
%
% This function is a commented/refactored version of the original MP_HT.m.
% The goal is readability WITHOUT changing any numerical results.
%
% Inputs
%   RI           : RI tomogram (Y x X x Z)
%   NAc, NAo     : condenser/objective NA
%   ResolutionX  : lateral voxel size (same unit as wavelength)
%   ResolutionZ  : axial voxel size
%   RImedium     : refractive index of the medium
%   wavelength   : illumination wavelength
%
% Output
%   RI_filtered_log : log10(MP-HT response), cropped to remove z-padding
%
% Notes
%   - Uses a symmetric z-padding to mitigate boundary artifacts.
%   - GPU acceleration is enabled by default (useGPU = true).
% -------------------------------------------------------------------------

useGPU = true;

% Symmetric padding in Z (keep identical to the original implementation)
RI = cat(3, RI(:,:,4:-1:2), RI, RI(:,:,end-1:-1:end-3));

% Spatial frequency support (2D) used to estimate band-pass parameters
otfSupport2D = OTF2(RI, ResolutionX, ResolutionZ, NAc, NAo, wavelength, RImedium);

% Apply MP-HT filtering (3D band-pass + squaring + Gaussian smoothing)
RI_filtered = mp_ht(single(RI), otfSupport2D, ResolutionX, ResolutionZ, useGPU);

% Remove the z-padding and take log10
RI_filtered_log = log10(RI_filtered(:,:,4:end-3));

end


% ========================================================================
% Helper: 2D spatial frequency support (Ewald overlap)
% ========================================================================
function res = OTF2(RI, dx, dz, NAc, NAo, wavelength, n_m)
%OTF2  Compute a 2D overlap support in (ux, uz) for parameter extraction.
%
% RI is only used for size; values do not affect this function.

Lx = size(RI,2) * dx;
Lz = size(RI,3) * dz;

dux = 1 / Lx;
duz = 1 / Lz;

ux = (1:size(RI,2)) - ceil((size(RI,2)+1)/2);
uz = (1:size(RI,3)) - ceil((size(RI,3)+1)/2);

ux = dux * reshape(ux, 1, []);
uz = duz * reshape(uz, [], 1);

% Ewald spheres (coherent illumination NA and objective NA)
Ewaldc = round((sqrt((n_m/wavelength).^2 - ux.^2) - (n_m/wavelength) - uz) / duz) == 0;
Ewaldc = Ewaldc .* ((ux.^2) < (NAc/wavelength)^2);

Ewaldo = round((sqrt((n_m/wavelength).^2 - ux.^2) - (n_m/wavelength) - uz) / duz) == 0;
Ewaldo = Ewaldo .* ((ux.^2) < (NAo/wavelength)^2);

% Overlap support
Ewaldc = fftn(Ewaldc);
Ewaldo = fftn(Ewaldo);
res = abs(ifftn(Ewaldc .* conj(Ewaldo)));
res = res > 0.5;
end


% ========================================================================
% Helper: MP-HT core filtering
% ========================================================================
function [filtered, RI] = mp_ht(RI, otf, dx, dz, use_gpu)
%MP_HT  Core MP-HT filtering.
%
% Steps (kept identical):
%   1) Symmetrize/support-clean the 2D OTF mask
%   2) Estimate torus-like band-pass parameters (alpha, beta, urc)
%   3) Build a 3D filter in (ur, uz)
%   4) Apply filter in Fourier domain, take magnitude^2
%   5) Gaussian smoothing (XY)

% ---- 1) Post-process OTF support (2D) ----
otf = otf + circshift(flip(otf,1), [1 0]);
otf = fftshift(otf);
otf(otf > 0) = 1;
otf(:,1:floor(size(otf,2)/2)) = 0;
otf = imclose(otf, strel('disk', 3));

% ---- 2) Frequency grids ----
Lx = size(RI,2) * dx;
Ly = size(RI,1) * dx;
Lz = size(RI,3) * dz;

dux = 1 / Lx;
duy = 1 / Ly;
duz = 1 / Lz;

ux = (1:size(RI,2)) - ceil((size(RI,2)+1)/2);
uy = (1:size(RI,1)) - ceil((size(RI,1)+1)/2);
uz = (1:size(RI,3)) - ceil((size(RI,3)+1)/2);

ux = dux * reshape(ux, 1, [], 1);
uy = duy * reshape(uy, [], 1, 1);
uz = duz * reshape(uz, 1, 1, []);
ur = sqrt(ux.^2 + uy.^2);

% ---- 3) Extract band-pass parameters from the 2D OTF mask ----
umax = max(sum(otf, 1));
urc = sum(ux .* (sum(otf,1) == umax) .* (ux > 0)) / sum((sum(otf,1) == umax) .* (ux > 0));

umin = min(abs(sum(otf,1) - umax/2));
ura = sum(ux .* (abs(sum(otf,1) - umax/2) == umin) .* ((ux > 0) & (ux < urc))) / ...
      sum((abs(sum(otf,1) - umax/2) == umin) .* ((ux > 0) & (ux < urc)));

alpha = umax * duz / 2;
beta  = (urc - ura);

% Sanity check: does the analytic filter extend beyond the measured support?
filter = 1 - ((ux - urc) / beta).^2 - (uz).^2 / alpha^2 > 0;
if sum(squeeze(filter).' - otf > 0, 'all') > 0
    disp("Filter goes beyond OTF")
end

% ---- 4) Build final 3D filter (nonnegative) ----
ur  = single(ur);
urc = single(urc);
uz  = single(uz);

filter = max(1 - (ur - urc).^2 / beta^2 - uz.^2 / alpha^2, 0);

% ---- 5) Apply filter in Fourier domain ----
if use_gpu
    filter = gpuArray(single(filter));
    RI = gpuArray(single(RI));
end

filter = ifftshift(filter);
RI = abs(ifftn(fftn(RI) .* filter)).^2;

% Gaussian smoothing in XY (same call as original; keep sigma expression)
filtered = imgaussfilt(RI, [1/(8*alpha*dx) 1/(8*alpha*dx)]);

if use_gpu
    RI = gather(RI);
    filtered = gather(filtered);
end

end
