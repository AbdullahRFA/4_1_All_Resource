image = imread('flower.jpeg');
gray = rgb2gray(image);
edges = edge(gray, 'Canny');
[H, T, R] = hough(edges);
P = houghpeaks(H, 5, 'threshold', ceil(0.3 * max(H(:))));
lines = houghlines(edges, T, R, P, 'FillGap', 20, 'MinLength', 40);
figure, imshow(image), hold on
for k = 1:length(lines)
    xy = [lines(k).point1; lines(k).point2];
    plot(xy(:, 1), xy(:, 2), 'LineWidth', 2, 'Color', 'red');
    plot(xy(1, 1), xy(1, 2), 'x', 'LineWidth', 2, 'Color', 'yellow');
    plot(xy(2, 1), xy(2, 2), 'x', 'LineWidth', 2, 'Color', 'green');
end
title('Detected Lines using Hough Transform');
hold off;