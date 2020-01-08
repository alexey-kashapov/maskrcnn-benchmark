from .my_dataset_eval import do_my_dataset_evaluation
from maskrcnn_benchmark.data.datasets import MyDataset


def my_dataset_evaluation(
    dataset,
    predictions,
    output_folder,
    box_only,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
):
    if isinstance(dataset, MyDataset):
        return do_my_dataset_evaluation(
            dataset=dataset,
            predictions=predictions,
            box_only=box_only,
            output_folder=output_folder,
            iou_types=iou_types,
            expected_results=expected_results,
            expected_results_sigma_tol=expected_results_sigma_tol,
        )
    else:
        raise NotImplementedError(
            (
                "Ground truth dataset is not a COCODataset, "
                "nor it is derived from AbstractDataset: type(dataset)="
                "%s" % type(dataset)
            )
        )
