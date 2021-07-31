import torch

debug_opt = False

def _train(
    model,
    dataloader,
    crit,
    opt,
    device,
    stat,
    vis,
    step,
):
    model.train()
    tot_loss = 0
    for i, data in enumerate(dataloader):
        input, digit, _ = data
        input, digit = input.to(device), digit.long().to(device)

        if debug_opt:
            print(input.shape, digit.shape)
        output = model(input)

        if debug_opt:
            print(output.shape, digit.shape)
        
        loss = crit(output, digit)
        tot_loss+= loss.detach().cpu()*len(digit)
        opt.zero_grad()
        loss.backward()
        opt.step()

        stat.step(output, digit, loss)
        vis.visline.plot(
            var_name= "train_loss_step",
            split_name= "train_loss",
            title_name= "loss per step",
            x = torch.tensor(step),
            y = loss.detach().cpu(),
        )
        
        step+=1

    return tot_loss, step
    