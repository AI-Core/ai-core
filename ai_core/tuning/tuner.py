from ray import tune
import os
import torch

# def tune_that_shit():


def tuner(trainable, tunable_params, num_samples=20):
    # print(config)

    scheduler = tune.schedulers.ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=10,
        grace_period=1,
        reduction_factor=2
    )

    reporter = tune.CLIReporter(
        parameter_columns=list(tunable_params.keys()),
        metric_columns=["loss", "training_iteration"]
    )

    resources = {"cpu": 1, "gpu": 0.2} if torch.cuda.is_available() else {'cpu': 1}

    result = tune.run(
        trainable,
        resources_per_trial={"cpu": 1, "gpu": 0},
        config=tunable_params,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        raise_on_failed_trial=False
    )


    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    # print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

    # best_trained_model = NN(best_trial.config["n_layers"])
    # device = "cpu"
    # if torch.cuda.is_available():
    #     device = "cuda:0"
    #     if gpus_per_trial > 1:
    #         best_trained_model = nn.DataParallel(best_trained_model)
    # best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    best_checkpoint_save = os.path.join(best_checkpoint_dir, "checkpoint")
    print(f'best checkpoint found at {best_checkpoint_save}')
    # model_state, optimizer_state = torch.load(best_checkpoint_save)
    # best_trained_model.load_state_dict(model_state)

    return result





# def test_accuracy(net, device="cpu"):
   
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data in test_loader:
#             images, labels = data
#             images, labels = images.to(device), labels.to(device)
#             outputs = net(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     return correct / total

# test_acc = test_accuracy(best_trained_model, device)
# print(f"Best trial test set accuracy: ({test_acc*100}%) achieved with {best_trial.config['n_layers']} layers")