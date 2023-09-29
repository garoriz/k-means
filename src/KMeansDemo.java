import javax.swing.*;
import java.awt.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

class Point {
    double x;
    double y;

    Point(double x, double y) {
        this.x = x;
        this.y = y;
    }
}

class KMeans {
    private List<Point> data;
    private int k;
    private List<Point> centroids;
    private List<List<Point>> clusters;

    KMeans(List<Point> data, int k) {
        this.data = data;
        this.k = k;
        this.centroids = initializeCentroids();
    }

    private List<Point> initializeCentroids() {
        Point pointCntr = new Point(0.0, 0.0);
        for (Point datum : data) {
            pointCntr.x += datum.x;
            pointCntr.y += datum.y;
        }
        pointCntr.x /= data.size();
        pointCntr.y /= data.size();
        double R = 0;
        for (Point datum : data) {
            double d = distance(datum, pointCntr);
            if (d > R) {
                R = d;
            }
        }
        List<Point> initialCentroids = new ArrayList<>();

        for (int i = 0; i < k; i++) {
            initialCentroids.add(new Point(R * Math.cos(2 * Math.PI * i / k) + pointCntr.x,
                    R * Math.sin(2 * Math.PI * i / k) + pointCntr.y));
        }

        return initialCentroids;
    }

    private double distance(Point point1, Point point2) {
        double dx = point1.x - point2.x;
        double dy = point1.y - point2.y;
        return Math.sqrt(dx * dx + dy * dy);
    }

    private int findClosestCentroid(Point point) {
        int closestCentroidIdx = -1;
        double minDistance = Double.MAX_VALUE;

        for (int i = 0; i < centroids.size(); i++) {
            double distance = distance(point, centroids.get(i));
            if (distance < minDistance) {
                minDistance = distance;
                closestCentroidIdx = i;
            }
        }

        return closestCentroidIdx;
    }

    private void assignToClusters() {
        clusters = new ArrayList<>(k);

        for (int i = 0; i < k; i++) {
            clusters.add(new ArrayList<>());
        }

        for (Point point : data) {
            int closestCentroidIdx = findClosestCentroid(point);
            clusters.get(closestCentroidIdx).add(point);
        }
    }

    private Point calculateCentroid(List<Point> cluster) {
        double sumX = 0;
        double sumY = 0;

        for (Point point : cluster) {
            sumX += point.x;
            sumY += point.y;
        }

        int clusterSize = cluster.size();
        return new Point(sumX / clusterSize, sumY / clusterSize);
    }

    private boolean updateCentroids() {
        List<Point> newCentroids = new ArrayList<>(k);
        boolean hasChanged = false;

        for (int i = 0; i < k; i++) {
            Point centroid = calculateCentroid(clusters.get(i));
            newCentroids.add(centroid);

            if (!centroid.equals(centroids.get(i))) {
                hasChanged = true;
            }
        }

        centroids = newCentroids;
        return hasChanged;
    }

    public List<Point> run(int maxIterations) {
        for (int iteration = 0; iteration < maxIterations; iteration++) {
            assignToClusters();
            if (!updateCentroids()) {
                break;
            }
        }

        return centroids;
    }

    public List<List<Point>> getClusters() {
        return clusters;
    }
}

class ClusterPlot extends JPanel {
    private List<Point> data;
    private List<List<Point>> clusters;
    private List<Point> centroids;

    ClusterPlot(List<Point> data, List<Point> centroids, List<List<Point>> clusters) {
        this.data = data;
        this.centroids = centroids;
        this.clusters = clusters;
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);

        // Отрисовка точек данных
        for (Point point : data) {
            int x = (int) (point.x * getWidth());
            int y = (int) (point.y * getHeight());
            g.setColor(Color.BLACK);
            g.fillOval(x, y, 5, 5);
        }

        // Отрисовка кластеров и центроидов
        String[] colors = {"#FF0000", "#0000FF", "#008000", "#FFA500", "#FF00FF"};
        for (int i = 0; i < clusters.size(); i++) {
            List<Point> cluster = clusters.get(i);
            Point centroid = centroids.get(i);
            g.setColor(Color.decode(colors[i % colors.length]));

            for (Point point : cluster) {
                int x = (int) (point.x * getWidth());
                int y = (int) (point.y * getHeight());
                g.fillOval(x, y, 5, 5);
            }

            g.setColor(Color.BLACK);
            int x = (int) (centroid.x * getWidth());
            int y = (int) (centroid.y * getHeight());
            g.fillOval(x - 10, y - 10, 20, 20);
        }
    }
}

public class KMeansDemo {
    public static void main(String[] args) {
        // Генерация случайных данных для примера
        List<Point> data = new ArrayList<>();
        Random random = new Random();

        for (int i = 0; i < 100; i++) {
            double x = random.nextDouble();
            double y = random.nextDouble();
            data.add(new Point(x, y));
        }

        // Нахождение оптимального количества кластеров
        int maxK = 10;
        int optimalK = findOptimalK(data, maxK);
        System.out.println("Оптимальное количество кластеров: " + optimalK);

        // Выполнение кластеризации с оптимальным количеством кластеров
        KMeans kMeans = new KMeans(data, optimalK);
        List<Point> centroids = kMeans.run(100);
        List<List<Point>> clusters = kMeans.getClusters();

        // Создание графического окна для отрисовки кластеров
        JFrame frame = new JFrame("K-Means Clustering");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        ClusterPlot clusterPlot = new ClusterPlot(data, centroids, clusters);
        frame.add(clusterPlot);
        frame.setSize(800, 600);
        frame.setVisible(true);
    }

    private static int findOptimalK(List<Point> data, int maxK) {
        List<Double> wcssValues = new ArrayList<>();

        for (int k = 1; k <= maxK; k++) {
            KMeans kMeans = new KMeans(data, k);
            List<Point> centroids = kMeans.run(100);
            List<List<Point>> clusters = kMeans.getClusters();
            double wcss = calculateWCSS(centroids, clusters);
            wcssValues.add(wcss);
        }

        // Нахождение оптимального количества кластеров
        int optimalK = 1;
        double minDiff = Double.MAX_VALUE;

        for (int i = 1; i < wcssValues.size() - 1; i++) {
            double diff = Math.abs(wcssValues.get(i) - wcssValues.get(i + 1)) /
                    Math.abs(wcssValues.get(i - 1) - wcssValues.get(i));
            if (diff < minDiff) {
                minDiff = diff;
                optimalK = i;
            }
        }

        return optimalK;
    }

    private static double calculateWCSS(List<Point> centroids, List<List<Point>> clusters) {
        double wcss = 0.0;

        for (int i = 0; i < centroids.size(); i++) {
            Point centroid = centroids.get(i);
            List<Point> cluster = clusters.get(i);

            for (Point point : cluster) {
                wcss += Math.pow(point.x - centroid.x, 2) + Math.pow(point.y - centroid.y, 2);
            }
        }

        return wcss;
    }
}
