function showOrthoslices(volume, xc, yc, zc,zscale,range)
% showOrthoslices(volume, xc, yc, zc)
%   Display orthogonal slices (XY, XZ, YZ) from a 3D volume.
%   Supports grayscale (Y×X×Z) and RGB (Y×X×Z×3) volumes.
%
% INPUTS:
%   volume : 3D or 4D array (Y×X×Z or Y×X×Z×3)
%   xc, yc, zc : slice indices along X, Y, Z (1-based)

    if nargin < 6
        range = double([min(volume(:)) max(volume(:))]);
    end


    dims = size(volume);
    isRGB = (ndims(volume) == 4 && dims(4) == 3);
    ny = dims(1);
    nx = dims(2);
    nz = dims(3);
    % Clip indices
    xc = max(1, min(nx, round(xc)));
    yc = max(1, min(ny, round(yc)));
    zc = max(1, min(nz, round(zc)));
    % Extract slices
    if isRGB
        % RGB: Y×X×Z×3
        sliceXY = squeeze(volume(:,:,zc,:));      % Y×X×3
        sliceXZ = squeeze(volume(yc,:,:,:));      % X×Z×3
        sliceYZ = squeeze(volume(:,xc,:,:));      % Y×Z×3
        % Adjust orientation
        sliceXZ = permute(sliceXZ, [2 1 3]);      % Z×X×3
    else
        % Grayscale
        sliceXY = squeeze(volume(:,:,zc));        % Y×X
        sliceXZ = squeeze(volume(yc,:,:));        % X×Z
        sliceYZ = squeeze(volume(:,xc,:));        % Y×Z
        % Transpose for orientation
        sliceXZ = sliceXZ';                       % Z×X
    end
    sliceXZ = imresize(sliceXZ,METHOD = 'bilinear',Scale = [zscale 1]);
    sliceYZ = imresize(sliceYZ,METHOD = 'bilinear',Scale = [1 zscale]);
    % Plot
    figure;
    colormap gray;
    % ---- XY plane ----
    subplot(1,3,1);
    if isRGB
        imshow(sliceXY);
        axis image;
        hold on
        plot([1 nx],[yc yc],'-w')
        plot([xc xc],[1 ny],'-w')
        hold off
    else
        imagesc(sliceXY);
        axis image;
        hold on
        plot([1 nx],[yc yc],'-w')
        plot([xc xc],[1 ny],'-w')
        hold off
        colorbar;
        clim(range)
    end
    title(sprintf('XY @ Z=%d', zc));
    xlabel('X'); ylabel('Y');
    % ---- XZ plane ----
    subplot(1,3,2);
    if isRGB
        imshow(sliceXZ);
        axis image;
        hold on
        plot([xc xc],[1 size(sliceXZ,1)],'-w')
        plot([1 nx],[zc*zscale zc*zscale],'-w')
        hold off
    else
        imagesc(sliceXZ);
        axis image;
        hold on
        plot([xc xc],[1 size(sliceXZ,1)],'-w')
        plot([1 nx],[zc*zscale zc*zscale],'-w')
        hold off
        colorbar;
        clim(range)
    end
    title(sprintf('XZ @ Y=%d', yc));
    xlabel('X'); ylabel('Z');
    % ---- YZ plane ----
    subplot(1,3,3);
    if isRGB
        imshow(sliceYZ);
        axis image;
        hold on
        plot([1 size(sliceYZ,2)],[yc yc],'-w')
        plot([zc*zscale zc*zscale],[1 ny],'-w')
        hold off
    else
        imagesc(sliceYZ);
        axis image;
        hold on
        plot([1 size(sliceYZ,2)],[yc yc],'-w')
        plot([zc*zscale zc*zscale],[1 ny],'-w')
        hold off
        colorbar;
        clim(range)
    end
    title(sprintf('YZ @ X=%d', xc));
    xlabel('Z'); ylabel('Y');
end