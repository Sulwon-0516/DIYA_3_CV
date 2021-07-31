import torch

debug_opt = False

def _val(
    model,
    dataloader,
    crit,
    device,
    stat,
):
    model.eval()
    with torch.no_grad():
        tot_loss = 0
        for i, data in enumerate(dataloader):
            input, digit = data
            input, digit = input.to(device), digit.long().to(device)

            output = model(input)

            loss = crit(output, digit)
            tot_loss+= loss.detach().cpu()*len(digit)
            
            stat.step(output, digit, loss)
            
        