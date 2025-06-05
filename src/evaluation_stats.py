import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.evaluation import evaluation_readingTest


# The full text of the readingTestFluencE
ground_truth = 'sɛ listwaʁ də møsjø pəti ki vi dɑ˜z yn vjɛj mɛzɔ˜ sitye o kœʁ dœ˜ vjø vilaʒ la mɛzɔ˜ ɛt ɑ˜tuʁe dœ˜ ʒaʁdɛ˜ avɛk yn baʁjɛʁ il i a de kɔ˜kɔ˜bʁ de ʃu fʁize tut sɔʁt də leɡymz o fɔ˜ dy ʒaʁdɛ˜ lə pɔʁtijɔ˜ ʁɛst tuʒuʁ fɛʁme puʁ kə ʃjɛ˜ a pys nə seʃap pa ʃjɛ˜ a pysz ɛm sə kuʃe pʁɛ də la pubɛl a lɔ˜bʁ dœ˜n ɔʁɑ˜ʒe kuvɛʁ də fʁyi delisjø ʃjɛ˜ a pysz ɛ ɡuʁmɑ˜ il kʁɔk tu sə ki lyi pas su la dɑ˜ dez ɔʁɑ˜ʒ puʁi ki tɔ˜b syʁ lə sɔl de flœʁ fanez œ˜ mɔʁso də byvaʁ œ˜ ʒuʁ məsjø pəti desid də mɛtʁ ʃjɛ̃ a pys dɑ˜z yn niʃ ʃjɛ˜ a pys nɛm paz ɛtʁ ɑ˜fɛʁme il pʁefɛʁ sɑ˜dɔʁmiʁ ɑ˜ ʁəɡaʁdɑ˜ lez etwal dɑ˜ lə sjɛl tut le nyiz il abwa kɑ˜ məsjø pəti va sə kuʃe məsjø pəti desid də dɔʁmiʁ dɑ˜ lə ɡʁənje də sa ʒɔli mɛzɔ˜ puʁ pʁɑ˜dʁ œ˜ pø də ʁəpoz il nə tʁuv ply lə sɔmɛj yn nyi dɛ˜sɔmni ɔp il sot dy li e uvʁ la ɡʁɑ˜d mal ki sə tʁuv dəvɑ˜ lyi dɑ˜z œ˜ kwɛ˜ sɔ˜bʁ dy ɡʁənje e la syʁpʁiz tut sa vi kil pɑ˜sɛ sɑ˜z istwaʁ lyi ʁəvjɛ˜t ɑ˜ memwaʁ il sɔʁ le muʃwaʁ bʁɔde paʁ sa ɡʁɑ˜mɛʁ se pətit dɑ˜ də lɛ sɔ˜ po də ʃɑ˜bʁ ebʁeʃe yn tɛt də pwasɔ˜ seʃe œ˜ sak plɛ˜ də bijz yn mɔ˜tʁ ki fɛ tik tak tik tak sɔ˜ kaʁnɛ də nɔtz œ˜ bu də lasɛ sɔ˜ vjø tʁɑ˜zistɔʁ a pil sɛ fu kɔm tu se suvniʁ sə buskyl dɑ˜ sa tɛt e il nə pø ʁətniʁ se laʁm demɔsjɔ˜ sa vi nɛ pa sɑ˜z istwaʁ il sə suvjɛ˜t ɛɡzaktəmɑ̃ də la vwa dy pʁezɑ˜tatœʁ meteo lə tɑ˜ va sameljɔʁe dəmɛ˜ ɑ˜ deby də matine syʁ nɔtʁ ʁeʒjɔ˜ sjɛl ʃaʁʒe lapʁɛmidi il sə ʁapɛl le vjɛj pyblisitez aɛma e la salte sɑ̃ va ɔ˜n a tuʒuʁ bəzwɛ˜ də pəti pwa ʃe swa le pʁəmjɛʁ lymjɛʁ dy ʒuʁ penɛtʁ paʁ la pətit fənɛtʁ dy ɡʁənje il ɛt o kœʁ də se suvniʁ kɑ˜ sɔ˜ ʁevɛj sɔndʁɪŋ dʁɪŋ dʁɪŋ'.split(" ")

def run_all_evaluations(test_df, model, test_type, correct_threshold=0.3, incorrect_threshold=0.4):
    """
    This function runs the evaluation for all tests in the test_df DataFrame for a given model.

    Args:
        test_df (pd.DataFrame): DataFrame containing the test data.
        model (str): The model name to evaluate.
        test_type (str): The type of test (e.g., 'readingTestFluencE').
        correct_threshold (float): The threshold for correct predictions.
        incorrect_threshold (float): The threshold for incorrect predictions.
    
    Returns:
        pd.DataFrame: DataFrame containing the evaluation results for all tests.
    """
    # We create a dataframe that will contain the results of the evaluation
    results_df = pd.DataFrame(columns=['test_id', 'model', 'false_pos', 'false_neg', 'word_results'])

    # The paths to the folder containing the CSV files for the current model
    csv_folder_path =  f'transcriptions/{test_type}/{model}'

    # We extract all the test id in the test_df DataFrame
    tests_id = test_df['id'].unique()

    # For each test, we compute the evaluation results for the three models
    for test_id in tests_id:
        csv_file_path = f'{csv_folder_path}/{test_type}_{test_id}_phonemes.csv'

        # We run the evaluation function 
        word_results, false_pos, false_neg = evaluation_readingTest(test_df, csv_file_path, test_id, ground_truth, correct_threshold, incorrect_threshold)

        # Create new row as a DataFrame
        new_row = pd.DataFrame([{
            'test_id': test_id,
            'model': model,
            'false_pos': false_pos,
            'false_neg': false_neg,
            'word_results': word_results
        }])

        # Concatenate with existing results
        results_df = pd.concat([results_df, new_row], ignore_index=True)

    return results_df

def grid_search_evaluation(test_df, model, test_type):
    """
    This function performs a grid search over the thresholds for correct and incorrect predictions
    for a given model and test type.

    Args:
        test_df (pd.DataFrame): DataFrame containing the test data.
        model (str): The model name to evaluate.
        test_type (str): The type of test (e.g., 'readingTestFluencE').

    Returns:
        pd.DataFrame: DataFrame containing the evaluation results for all combinations of thresholds.
    """
    correct_thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    incorrect_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    # We create a dataframe that will contain the results of the evaluation
    results_df = pd.DataFrame(columns=['test_id', 'model', 'correct_threshold', 'incorrect_threshold', 'false_pos', 'false_neg', 'word_results'])

    # We iterate over all the possible combinations of correct and incorrect thresholds and compute the evaluation results
    for correct_th in correct_thresholds:
        for incorrect_th in incorrect_thresholds:
            # Run evaluation with current threshold combination
            eval_df = run_all_evaluations(
                test_df=test_df,
                model=model,
                test_type=test_type,
                correct_threshold=correct_th,
                incorrect_threshold=incorrect_th
            )

            # Add the threshold values to the result DataFrame
            eval_df['correct_threshold'] = correct_th
            eval_df['incorrect_threshold'] = incorrect_th

            # Append to the main results DataFrame
            results_df = pd.concat([results_df, eval_df], ignore_index=True)

    return results_df


def evaluate_and_plot_thresholds(test_df, model, test_type):
    """
    Runs grid search over thresholds, evaluates all test files, aggregates results,
    and plots heatmaps showing percentages of false positives, false negatives,
    and uncertain words.

    Args:
        test_df (pd.DataFrame): DataFrame containing test metadata and evaluations.
        model (str): Model name to evaluate.
        test_type (str): Test type (e.g. 'readingTestFluencE').

    Returns:
        pd.DataFrame: Aggregated results with columns
                      ['correct_threshold', 'incorrect_threshold', 'false_pos_pct', 'false_neg_pct', 'uncertain_pct']
    """
    results_list = []

    # Run grid search evaluations (reuse your existing function)
    all_results = grid_search_evaluation(test_df, model, test_type)

    # For each combination, aggregate over tests
    grouped = all_results.groupby(['correct_threshold', 'incorrect_threshold'])

    for (correct_th, incorrect_th), group in grouped:
        # Sum false pos/neg
        total_false_pos = group['false_pos'].sum()
        total_false_neg = group['false_neg'].sum()

        # Count uncertain words from word_results (word_results is a list of tuples)
        uncertain_count = 0
        total_words = 0

        for word_results in group['word_results']:
            counts = count_categories(word_results)
            uncertain_words = counts['uncertain_correct'] + counts['uncertain_incorrect']
            uncertain_count += uncertain_words
            total_words += counts.sum()

        # Calculate percentages
        false_pos_pct = 100 * total_false_pos / total_words if total_words else 0
        false_neg_pct = 100 * total_false_neg / total_words if total_words else 0
        uncertain_pct = 100 * uncertain_count / total_words if total_words else 0

        results_list.append({
            'correct_threshold': correct_th,
            'incorrect_threshold': incorrect_th,
            'false_pos_pct': false_pos_pct,
            'false_neg_pct': false_neg_pct,
            'uncertain_pct': uncertain_pct
        })

    results_df = pd.DataFrame(results_list)

    # Plotting heatmaps for visualization
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    for idx, metric in enumerate(['false_pos_pct', 'false_neg_pct', 'uncertain_pct']):
        pivot_table = results_df.pivot('correct_threshold', 'incorrect_threshold', metric)
        sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap='viridis', ax=axes[idx])
        axes[idx].set_title(metric.replace('_', ' ').capitalize())
        axes[idx].set_xlabel('Incorrect Threshold')
        axes[idx].set_ylabel('Correct Threshold')

    plt.suptitle(f"Evaluation metrics for model '{model}' on test '{test_type}'")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    return results_df


def count_categories(word_results):
    # If it's a string representation of a list, evaluate it
    if isinstance(word_results, str):
        try:
            word_results = eval(word_results)
        except:
            return pd.Series([0, 0, 0, 0], index=['correct', 'incorrect', 'uncertain_correct', 'uncertain_incorrect'])
    
    # Count each category
    counts = {'correct': 0, 'incorrect': 0, 'uncertain_correct': 0, 'uncertain_incorrect': 0}
    for entry in word_results:
        if len(entry) >= 2 and entry[1] in counts:
            counts[entry[1]] += 1
    return pd.Series(counts)


def find_best_threshold_combo(results_df, verbose=True):
    import pandas as pd

    # Rename columns to a consistent internal format
    df = results_df.rename(columns={
        'false_pos_pct': 'fp_pct',
        'false_neg_pct': 'fn_pct',
        'uncertain_pct': 'uncertain_pct'
    }).copy()

    # Apply Z-score normalization to each metric
    for col in ['fp_pct', 'fn_pct', 'uncertain_pct']:
        mean = df[col].mean()
        std = df[col].std()
        if std < 1e-8:  # Avoid division by zero
            df[col + '_norm'] = 0.0
        else:
            df[col + '_norm'] = (df[col] - mean) / std

    # Define weights for normalized metrics
    weights = {
        'fp_pct_norm': 0.2,       # Adjust based on your cost tradeoffs
        'fn_pct_norm': 0.4,
        'uncertain_pct_norm': 0.4
    }

    # Compute composite score from normalized values
    df['score'] = (
        weights['fp_pct_norm'] * df['fp_pct_norm'] +
        weights['fn_pct_norm'] * df['fn_pct_norm'] +
        weights['uncertain_pct_norm'] * df['uncertain_pct_norm']
    )

    # Select the row with the lowest score
    best_row = df.sort_values('score').iloc[0]

    if verbose:
        print("✅ Best threshold combination:")
        print(f"  - correct_threshold: {best_row['correct_threshold']}")
        print(f"  - incorrect_threshold: {best_row['incorrect_threshold']}")
        print(f"  - False Positives: {best_row['fp_pct']:.2f}%")
        print(f"  - False Negatives: {best_row['fn_pct']:.2f}%")
        print(f"  - Uncertain: {best_row['uncertain_pct']:.2f}%")
        print(f"  - Composite Score: {best_row['score']:.4f}")

    return best_row



def plot_word_classification(all_eval_df):
    # Define the categories to count
    categories = ['correct', 'uncertain_correct', 'incorrect', 'uncertain_incorrect']

    def count_classifications(word_results):
        # Count occurrences of each category in the word results
        counts = {category: 0 for category in categories}
        for word, classification, _ in word_results:
            if classification in counts:
                counts[classification] += 1
        return counts
    
    # Apply the counting function for each row and sum the counts
    all_eval_df['classification_counts'] = all_eval_df['word_results'].apply(count_classifications)

    # Expand classification counts into separate columns
    classification_df = pd.json_normalize(all_eval_df['classification_counts'])

    # Combine the original dataframe with classification counts
    df_combined = pd.concat([all_eval_df, classification_df], axis=1)

    # Group by model and sum the counts
    model_counts = df_combined.groupby('model')[categories].sum().reset_index()

    # Plot the results
    model_counts.set_index('model')[categories].plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title('Word Classifications by Model')
    plt.xlabel('Model')
    plt.ylabel('Number of Words')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_word_classification_trends(model_df, model):
    all_eval_df = model_df.copy()

    # Define the categories to count
    categories = ['correct', 'uncertain_correct', 'incorrect', 'uncertain_incorrect']

    # Define a helper function to count occurrences of each category
    def count_classifications(word_results):
        counts = {category: 0 for category in categories}
        for word, classification, _ in word_results:
            if classification in counts:
                counts[classification] += 1
        return counts

    # Apply the counting function for each row
    all_eval_df['classification_counts'] = all_eval_df['word_results'].apply(count_classifications)

    # Expand classification counts into separate columns
    classification_df = pd.json_normalize(all_eval_df['classification_counts'])

    # Combine with original dataframe
    df_combined = pd.concat([all_eval_df, classification_df], axis=1)

    # Compute mean values for each threshold and classification category
    mean_df = df_combined.groupby(['correct_threshold', 'incorrect_threshold'])[
        ['correct', 'uncertain_correct', 'incorrect', 'uncertain_incorrect']
    ].mean().reset_index()

    # Plot: correct vs correct_threshold
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=mean_df, x='correct_threshold', y='correct', marker='o', color='green')
    plt.title(f'{model} - Mean Correct Count vs Correct Threshold')
    plt.xlabel('Correct Threshold')
    plt.ylabel('Mean Count of Correct Words')
    plt.tight_layout()
    plt.show()

    # Plot: uncertain_correct vs correct_threshold
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=mean_df, x='correct_threshold', y='uncertain_correct', marker='o', color='orange')
    plt.title(f'{model} - Mean Uncertain Correct Count vs Correct Threshold')
    plt.xlabel('Correct Threshold')
    plt.ylabel('Mean Count of Uncertain Correct Words')
    plt.tight_layout()
    plt.show()

    # Plot: incorrect vs incorrect_threshold
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=mean_df, x='incorrect_threshold', y='incorrect', marker='o', color='red')
    plt.title(f'{model} - Mean Incorrect Count vs Incorrect Threshold')
    plt.xlabel('Incorrect Threshold')
    plt.ylabel('Mean Count of Incorrect Words')
    plt.tight_layout()
    plt.show()

    # Plot: uncertain_incorrect vs incorrect_threshold
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=mean_df, x='incorrect_threshold', y='uncertain_incorrect', marker='o', color='purple')
    plt.title(f'{model} - Mean Uncertain Incorrect Count vs Incorrect Threshold')
    plt.xlabel('Incorrect Threshold')
    plt.ylabel('Mean Count of Uncertain Incorrect Words')
    plt.tight_layout()
    plt.show()
    

def plot_FP_FN_stats(model_df, model):
    """
    Plot the false positive and false negative statistics for a given model.

    Args:
        model_df (pd.DataFrame): DataFrame containing the evaluation results for the model.
        model (str): The model name to evaluate.
    """
    # Pivot table for heatmap of total errors (FP + FN)
    total_error_pivot = model_df.pivot_table(
        index='correct_threshold',
        columns='incorrect_threshold',
        values='total_error',
        aggfunc='mean'  # Handle duplicates by taking the mean
    )

    # Pivot table for heatmap of false positives (original heatmap version)
    false_pos_pivot = model_df.pivot_table(
        index='correct_threshold',
        columns='incorrect_threshold',
        values='false_pos',
        aggfunc='mean'  # Handle duplicates by taking the mean
    )

    # Pivot table for heatmap of false negatives (original heatmap version)
    false_neg_pivot = model_df.pivot_table(
        index='correct_threshold',
        columns='incorrect_threshold',
        values='false_neg',
        aggfunc='mean'  # Handle duplicates by taking the mean
    )

    # Ensure data is numeric (convert to float and handle missing values)
    total_error_pivot = total_error_pivot.astype(float)
    false_pos_pivot = false_pos_pivot.astype(float)
    false_neg_pivot = false_neg_pivot.astype(float)

    # Plot for total errors (FP + FN) as heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        total_error_pivot,
        annot=True,
        fmt=".2f",  # Display with 2 decimal places
        cmap="Reds",
        cbar_kws={'label': 'Total Errors (FP + FN)'},
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title(f'Total Errors (FP + FN) - {model}')
    plt.xlabel("Incorrect Threshold")
    plt.ylabel("Correct Threshold")
    plt.tight_layout()
    plt.show()

    # Plot for false positives as a line plot (simplified)
    plt.figure(figsize=(8, 6))
    sns.lineplot(
        data=model_df,
        x='correct_threshold',
        y='false_pos',
        marker='o',
        color='red'
    )
    plt.title(f'False Positives - {model}')
    plt.xlabel("Correct Threshold")
    plt.ylabel("False Positives")
    plt.tight_layout()
    plt.show()

    # Plot for false negatives as a line plot (simplified)
    plt.figure(figsize=(8, 6))
    sns.lineplot(
        data=model_df,
        x='incorrect_threshold',
        y='false_neg',
        marker='o',
        color='blue'
    )
    plt.title(f'False Negatives - {model}')
    plt.xlabel("Incorrect Threshold")
    plt.ylabel("False Negatives")
    plt.tight_layout()
    plt.show()