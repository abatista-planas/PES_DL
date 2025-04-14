"""collect some samples"""


def generate_sample(sample_number, base):
    print("Generating sample 1.1.0")
    return [base + sample_number % 3] * 5


def collect_training_samples(n_samples):
    return collect_samples(n_samples, 1)


def collect_testing_samples(n_samples):
    return collect_samples(n_samples, 0.5)


def collect_samples(n_samples, base):
    return [generate_sample(i, base) for i in range(n_samples)]