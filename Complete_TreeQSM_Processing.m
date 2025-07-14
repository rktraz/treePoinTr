%% Complete TreeQSM Processing Script
% This script takes a disconnected tree PLY file and creates a fully connected version
% Author: Assistant
% Date: 2025

clear; clc; close all;

%% =======================================================================
%% CONFIGURATION - MODIFY THESE SETTINGS
%% =======================================================================

% Input file path - CHANGE THIS to your PLY file path
input_filename = '\\files.math.uwaterloo.ca\rkaharly\ResearchDocuments\THESIS\231_branches_classsified_postprocessed_enhanced\231_branches_postprocessed.ply';

% Output file names
output_connected_ply = 'tree_connected_by_TreeQSM.ply';
output_centered_ply = 'tree_connected_centered.ply';
output_results_mat = 'TreeQSM_Complete_Results.mat';

% Point cloud generation settings
points_per_cylinder = 50;  % Higher = denser output (50-200 recommended)
max_input_points = 50000;  % Downsample if input is larger

%% =======================================================================
%% STEP 1: SETUP AND VALIDATION
%% =======================================================================

fprintf('🌳 TreeQSM Complete Processing Script\n');
fprintf('=====================================\n\n');

% Check if TreeQSM is available
if exist('treeqsm', 'file') ~= 2
    error('❌ TreeQSM not found in MATLAB path. Please install and add TreeQSM to path.');
end
fprintf('✅ TreeQSM found in MATLAB path\n');

% Check if input file exists
if ~exist(input_filename, 'file')
    error('❌ Input file not found: %s', input_filename);
end
fprintf('✅ Input file found: %s\n', input_filename);

%% =======================================================================
%% STEP 2: LOAD POINT CLOUD
%% =======================================================================

fprintf('\n📁 Loading point cloud...\n');
try
    ptCloud = pcread(input_filename);
    P_original = ptCloud.Location;
    fprintf('✅ Point cloud loaded successfully!\n');
    fprintf('   Number of points: %d\n', size(P_original, 1));
    fprintf('   Original bounds:\n');
    fprintf('     X: [%.2f, %.2f] m\n', min(P_original(:,1)), max(P_original(:,1)));
    fprintf('     Y: [%.2f, %.2f] m\n', min(P_original(:,2)), max(P_original(:,2)));
    fprintf('     Z: [%.2f, %.2f] m\n', min(P_original(:,3)), max(P_original(:,3)));
catch ME
    error('❌ Failed to load PLY file: %s', ME.message);
end

%% =======================================================================
%% STEP 3: PREPROCESS POINT CLOUD
%% =======================================================================

fprintf('\n🔧 Preprocessing point cloud...\n');

% Center the point cloud to avoid numerical issues
P_center = mean(P_original, 1);
P = P_original - P_center;
fprintf('✅ Point cloud centered to origin\n');

% Remove outliers
distances = sqrt(sum(P.^2, 2));
percentile_99 = prctile(distances, 99);
outlier_mask = distances <= percentile_99;
P_clean = P(outlier_mask, :);

removed_outliers = sum(~outlier_mask);
if removed_outliers > 0
    fprintf('✅ Removed %d outlier points\n', removed_outliers);
end

% Downsample if too large
if size(P_clean, 1) > max_input_points
    fprintf('⚠️  Large point cloud detected. Downsampling to %d points...\n', max_input_points);
    indices = randperm(size(P_clean, 1), max_input_points);
    P_final = P_clean(indices, :);
    fprintf('✅ Downsampled from %d to %d points\n', size(P_clean, 1), size(P_final, 1));
else
    P_final = P_clean;
end

fprintf('✅ Final preprocessed point cloud: %d points\n', size(P_final, 1));
fprintf('   Centered bounds:\n');
fprintf('     X: [%.2f, %.2f] m\n', min(P_final(:,1)), max(P_final(:,1)));
fprintf('     Y: [%.2f, %.2f] m\n', min(P_final(:,2)), max(P_final(:,2)));
fprintf('     Z: [%.2f, %.2f] m\n', min(P_final(:,3)), max(P_final(:,3)));

%% =======================================================================
%% STEP 4: RUN TREEQSM RECONSTRUCTION
%% =======================================================================

fprintf('\n🧠 Running TreeQSM reconstruction...\n');
fprintf('This may take several minutes depending on tree complexity...\n\n');

try
    % Define inputs
    inputs = define_input(P_final, 1, 1, 1);
    
    % Run TreeQSM
    QSM = treeqsm(P_final, inputs);
    
    fprintf('\n✅ TreeQSM reconstruction completed successfully!\n');
    
    % Display tree statistics
    fprintf('\n🌳 Tree Analysis Results:\n');
    fprintf('========================\n');
    fprintf('Number of cylinders: %d\n', length(QSM.cylinder.radius));
    fprintf('Number of branches: %d\n', length(QSM.branch.order));
    fprintf('Maximum branch order: %d\n', max(QSM.branch.order));
    fprintf('Total tree volume: %.1f L\n', sum(QSM.cylinder.volume) * 1000);
    fprintf('Tree height: %.2f m\n', max(QSM.cylinder.end(:,3)) - min(QSM.cylinder.start(:,3)));
    fprintf('Trunk diameter (DBH): %.3f m\n', mean([QSM.treedata.DBHqsm, QSM.treedata.DBHcyl]));
    
    % Branch order breakdown
    fprintf('\nBranch Order Distribution:\n');
    for order = 1:max(QSM.branch.order)
        count = sum(QSM.branch.order == order);
        fprintf('  Order %d: %d branches\n', order, count);
    end
    
catch ME
    fprintf('❌ TreeQSM reconstruction failed: %s\n', ME.message);
    fprintf('\nTroubleshooting suggestions:\n');
    fprintf('1. Check if point cloud represents a single tree\n');
    fprintf('2. Try reducing max_input_points to 20000\n');
    fprintf('3. Check for extremely sparse or dense regions\n');
    error('TreeQSM reconstruction failed');
end

%% =======================================================================
%% STEP 5: CONVERT TO CONNECTED POINT CLOUD
%% =======================================================================

fprintf('\n🔄 Converting cylindrical model to connected point cloud...\n');

% Extract cylinder information
cylinders = QSM.cylinder;
n_cylinders = length(cylinders.radius);

fprintf('Processing %d cylinders...\n', n_cylinders);

% Initialize output
connected_points = [];
total_points_generated = 0;

% Process each cylinder
for i = 1:n_cylinders
    % Get cylinder parameters
    start_pt = cylinders.start(i, :);
    end_pt = cylinders.end(i, :);
    radius = cylinders.radius(i);
    
    % Skip degenerate cylinders
    if radius < 1e-6
        continue;
    end
    
    % Generate points for this cylinder
    cylinder_points = generate_cylinder_points(start_pt, end_pt, radius, points_per_cylinder);
    connected_points = [connected_points; cylinder_points];
    total_points_generated = total_points_generated + size(cylinder_points, 1);
    
    % Progress indicator
    if mod(i, 100) == 0 || i == n_cylinders
        fprintf('  Processed %d/%d cylinders (%.1f%%)\n', i, n_cylinders, 100*i/n_cylinders);
    end
end

fprintf('✅ Generated %d points from %d cylinders\n', total_points_generated, n_cylinders);

% Transform back to original coordinate system
connected_points_original = connected_points + P_center;

fprintf('✅ Transformed back to original coordinates\n');

%% =======================================================================
%% STEP 6: SAVE RESULTS
%% =======================================================================

fprintf('\n💾 Saving results...\n');

try
    % Save connected tree in original coordinates
    connected_ptCloud_original = pointCloud(connected_points_original);
    pcwrite(connected_ptCloud_original, output_connected_ply);
    fprintf('✅ Connected tree saved: %s\n', output_connected_ply);
    
    % Save centered version for reference
    connected_ptCloud_centered = pointCloud(connected_points);
    pcwrite(connected_ptCloud_centered, output_centered_ply);
    fprintf('✅ Centered version saved: %s\n', output_centered_ply);
    
    % Save complete MATLAB results
    save(output_results_mat, 'QSM', 'connected_points', 'connected_points_original', ...
         'P_original', 'P_center', 'P_final', 'inputs', '-v7.3');
    fprintf('✅ Complete results saved: %s\n', output_results_mat);
    
    fprintf('\n📊 Output File Summary:\n');
    fprintf('======================\n');
    fprintf('Original input: %d points\n', size(P_original, 1));
    fprintf('Connected output: %d points\n', size(connected_points_original, 1));
    fprintf('Compression ratio: %.1fx\n', size(connected_points_original, 1) / size(P_original, 1));
    
catch ME
    fprintf('❌ Error saving files: %s\n', ME.message);
end

%% =======================================================================
%% STEP 7: CREATE VISUALIZATION
%% =======================================================================

fprintf('\n📈 Creating visualization...\n');

try
    % Create comparison figure
    figure('Name', 'TreeQSM Processing Results', 'Position', [100, 100, 1800, 600]);
    
    % Original point cloud
    subplot(1, 3, 1);
    scatter3(P_final(:,1), P_final(:,2), P_final(:,3), 2, 'r.', 'MarkerFaceAlpha', 0.7);
    title('Original Disconnected Tree');
    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
    axis equal; grid on;
    
    % TreeQSM cylinder model
    subplot(1, 3, 2);
    plot_cylinder_model(QSM.cylinder);
    title('TreeQSM Cylinder Model');
    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
    axis equal; grid on;
    
    % Connected point cloud
    subplot(1, 3, 3);
    scatter3(connected_points(:,1), connected_points(:,2), connected_points(:,3), 2, 'g.', 'MarkerFaceAlpha', 0.7);
    title('Connected Point Cloud');
    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
    axis equal; grid on;
    
    % Save figure
    saveas(gcf, 'TreeQSM_Processing_Results.png');
    fprintf('✅ Visualization saved: TreeQSM_Processing_Results.png\n');
    
catch ME
    fprintf('⚠️  Visualization creation failed: %s\n', ME.message);
end

%% =======================================================================
%% STEP 8: FINAL SUMMARY
%% =======================================================================

fprintf('\n🎉 PROCESSING COMPLETE!\n');
fprintf('========================\n\n');
fprintf('Your disconnected tree has been successfully converted to a fully connected structure!\n\n');

fprintf('📁 Files Created:\n');
fprintf('  • %s - Main output (connected tree)\n', output_connected_ply);
fprintf('  • %s - Centered version\n', output_centered_ply);
fprintf('  • %s - Complete MATLAB data\n', output_results_mat);
fprintf('  • TreeQSM_Processing_Results.png - Visualization\n\n');

fprintf('🌳 Tree Properties:\n');
fprintf('  • Original points: %d\n', size(P_original, 1));
fprintf('  • Connected points: %d\n', size(connected_points_original, 1));
fprintf('  • Branches: %d (orders 1-%d)\n', length(QSM.branch.order), max(QSM.branch.order));
fprintf('  • Tree height: %.2f m\n', max(QSM.cylinder.end(:,3)) - min(QSM.cylinder.start(:,3)));
fprintf('  • Total volume: %.1f L\n', sum(QSM.cylinder.volume) * 1000);

fprintf('\n✅ Ready to use your connected tree: %s\n', output_connected_ply);

%% =======================================================================
%% HELPER FUNCTIONS
%% =======================================================================

function points = generate_cylinder_points(start_pt, end_pt, radius, num_points)
    % Generate points around a cylinder surface
    
    % Cylinder axis
    axis_vec = end_pt - start_pt;
    axis_length = norm(axis_vec);
    
    % Handle degenerate case
    if axis_length < 1e-6
        points = repmat(start_pt, min(num_points, 10), 1);
        return;
    end
    
    axis_unit = axis_vec / axis_length;
    
    % Create perpendicular vectors
    if abs(axis_unit(1)) < 0.9
        perp1 = cross(axis_unit, [1, 0, 0]);
    else
        perp1 = cross(axis_unit, [0, 1, 0]);
    end
    
    % Normalize perpendicular vectors
    perp1 = perp1 / norm(perp1);
    perp2 = cross(axis_unit, perp1);
    perp2 = perp2 / norm(perp2);
    
    % Generate points
    num_around = max(6, ceil(sqrt(num_points)));  % At least 6 points around
    num_along = max(3, ceil(num_points / num_around));  % At least 3 points along
    
    points = [];
    for i = 1:num_along
        % Position along cylinder axis
        t = (i-1) / max(1, num_along-1);
        center = start_pt + t * axis_vec;
        
        % Points around circumference
        for j = 1:num_around
            angle = 2 * pi * (j-1) / num_around;
            point = center + radius * (cos(angle) * perp1 + sin(angle) * perp2);
            points = [points; point];
        end
    end
end 