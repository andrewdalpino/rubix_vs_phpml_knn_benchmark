<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Classifiers\KDNeighbors;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use Rubix\ML\Transformers\NumericStringConverter;
use League\Csv\Reader;

$dataset = Labeled::fromIterator(new CSV('banknote.csv', true))
    ->apply(new NumericStringConverter());

$dataset->transformLabels(function($label) {
    return $label === 1 ? 'conterfeit' : 'normal';
});

$dataset->toCSV()->write('banknote.new.csv');

[$training, $testing] = $dataset->stratifiedSplit(0.8);

$estimator = new KDNeighbors(5);

$start = microtime(true);

$estimator->train($training);

$trainTime = microtime(true) - $start;

$start = microtime(true);

$predictions = $estimator->predict($testing);

$inferTime = microtime(true) - $start;

$metric = new Accuracy();

$accuracy = $metric->score($predictions, $testing->labels());

echo "Training time: {$trainTime}s" . PHP_EOL;
echo "Infer time: {$inferTime}s" . PHP_EOL;
echo "Accuracy: $accuracy" . PHP_EOL;