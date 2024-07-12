from datetime import datetime

from torch.utils.tensorboard.writer import SummaryWriter


def print_similar(model, corpus, query_id, k=10):
    """
         Print the most similar words given a query
    """
    cos, dist = model.get_metrics(query_id)
    cos, dist = cos.cpu(), dist.cpu()
    vals, idxs = cos.topk(k)

    for val, idx in zip(vals, idxs):
        print(f"{corpus.id2word[idx.item()]:<20}:{val.item():.3f}")


def tfb_projector(model, corpus, log_dir):
    """
        Save the projector model to TensorBoard log_dir
    """
    now = datetime.now()
    log_dir = f'{log_dir}_{now.strftime("%Y_%m_%d-%H_%M_%S")}'
    writer = SummaryWriter(log_dir)
    embeddings = model.vEmbedding.weight
    labels = list(corpus.id2word.values())
    writer.add_embedding(embeddings, metadata=labels)
    writer.flush()
    writer.close()
    return log_dir
