from ray import tune
import os
import torch

# def tune_that_shit():


def tuner(train, tunable_params):
    # print(config)

    # def wrapper(*args, config):


        # def wrapit(train):
        #     def wrapper(config):
        #         train(fixed_params, **tunable_params)
        #     return wrapper

        # @wrapit
        # def tune_that_shit(fixed_params, tunable_params):
        #     train()

            

        

        def checkpoint():
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((model.state_dict(), optimiser.state_dict()), path)

        # tune.report(
        #     loss=val_loss / len(val_loader),
        #     accuracy=correct / (len(val_loader) * batch_size)
        # )

        scheduler = tune.schedulers.ASHAScheduler(
            metric="loss",
            mode="min",
            max_t=10,
            grace_period=1,
            reduction_factor=2
        )

        reporter = tune.CLIReporter(
            parameter_columns=list(tunable_params.keys()),
            metric_columns=["loss", "accuracy", "training_iteration"]
        )

        result = tune.run(
            train,
            resources_per_trial={"cpu": 2, "gpu": 0.2},
            config=tunable_params,
            num_samples=20,
            scheduler=scheduler,
            progress_reporter=reporter
        )
        return result




# best_trial = result.get_best_trial("loss", "min", "last")
# print("Best trial config: {}".format(best_trial.config))
# print("Best trial final validation loss: {}".format(
#     best_trial.last_result["loss"]))
# print("Best trial final validation accuracy: {}".format(
#     best_trial.last_result["accuracy"]))

# best_trained_model = NN(best_trial.config["n_layers"])
# device = "cpu"
# if torch.cuda.is_available():
#     device = "cuda:0"
#     if gpus_per_trial > 1:
#         best_trained_model = nn.DataParallel(best_trained_model)
# best_trained_model.to(device)

# best_checkpoint_dir = best_trial.checkpoint.value
# model_state, optimizer_state = torch.load(os.path.join(
#     best_checkpoint_dir, "checkpoint"))
# best_trained_model.load_state_dict(model_state)




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