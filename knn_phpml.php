<?php

include __DIR__ . '/vendor/autoload.php';

use Phpml\Classification\KNearestNeighbors;
use Phpml\Math\Distance\Euclidean;
use Phpml\CrossValidation\StratifiedRandomSplit;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use League\Csv\Reader;

$reader = Reader::createFromPath(__DIR__ . '/banknote.csv')
    ->setDelimiter(',')->setEnclosure('"')->setHeaderOffset(0);

$samples = $reader->getRecords([
    'variance', 'skewness', 'kurtosis', 'entropy',
]);

$labels = $reader->fetchColumn('class');

$samples = iterator_to_array($samples);
$labels = iterator_to_array($labels);

foreach ($samples as &$sample) {
    $sample = array_map('floatval', $sample);
}

$testingSamples = array_splice($samples, 0, (int) round(0.2 * count($samples)));
$testingLabels = array_splice($labels, 0, (int) round(0.2 * count($labels)));

$estimator = new KNearestNeighbors(5);

$start = microtime(true);

$estimator->train($samples, $labels);

$trainTime = microtime(true) - $start;

$start = microtime(true);

$predictions = $estimator->predict($testingSamples);

$inferTime = microtime(true) - $start;

$metric = new Accuracy();

$accuracy = $metric->score($predictions, $testingLabels);

echo "Training time: {$trainTime}s" . PHP_EOL;
echo "Infer time: {$inferTime}s" . PHP_EOL;
echo "Accuracy: $accuracy" . PHP_EOL;
