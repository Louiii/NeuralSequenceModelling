from data.classification.Translation import *
from _8_Attention.AttentionTypes import Encoder, Attention, RNN, LuongDecoder

PATH = '_8_Attention/models/'

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimiser,
                          decoder_optimiser, criterion, max_length=MAX_LENGTH):
    enc_h = encoder.initHidden()

    for p in encoder.parameters(): p.grad = None
    for p in decoder.parameters(): p.grad = None
    # encoder_optimiser.zero_grad()
    # decoder_optimiser.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    loss = 0
    enc_out, enc_h = encoder(input_tensor, enc_h)

    dec_inp = torch.tensor([[SOS_token]], device=device)
    dec_h = enc_h

    use_teacher_forcing = random.random() < teacher_forcing_ratio

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, dec_h, decoder_attention = decoder(dec_inp, dec_h, enc_out)
            
            loss += criterion(decoder_output[0], target_tensor[di])
            dec_inp = target_tensor[di].unsqueeze(0)  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, dec_h, decoder_attention = decoder(dec_inp, dec_h, enc_out)
            
            topv, topi = decoder_output.topk(1)
            dec_inp = topi.squeeze(0).detach()  # detach from history as input
            
            loss += criterion(decoder_output[0], target_tensor[di])
            if dec_inp.item() == EOS_token:
                break

    loss.backward()

    encoder_optimiser.step()
    decoder_optimiser.step()

    return loss.item() / target_length

def trainIters(encoder, decoder, encoder_optimiser, decoder_optimiser, n_iters, 
               input_lang, output_lang, device, path, print_every=1000, plot_every=100):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    training_pairs = [tensorsFromPair(random.choice(pairs), input_lang, output_lang, device)
                                                       for i in range(n_iters)]
    criterion = nn.NLLLoss()

    pbar = tqdm(range(1, n_iters + 1))
    for iter in pbar:
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimiser, decoder_optimiser, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            pbar.set_description('Loss: %.4f' % print_loss_avg)
            checkpoint_save(iter, encoder, decoder, encoder_optimiser,
                                decoder_optimiser, path)

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


#save_model(encoder1, attn_decoder1, 'models/')

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
    print(random.choice(pairs))
    

    teacher_forcing_ratio = 0.5
    attention_type='general'
    rnn_type = 'GRU'

    h_dim = 256
    bidir = False
    encoder = Encoder( input_lang.n_words, h_dim, rnn_type, bidir )#.to(device)

    attn = Attention(h_dim, h_dim, attention_type=attention_type, max_length=MAX_LENGTH)

    decoder = LuongDecoder(attn, h_dim, output_lang.n_words, rnn_type)

    learning_rate=0.01
    encoder_optimiser = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimiser = optim.SGD(decoder.parameters(), lr=learning_rate)

    # decoder = DecoderRNN( h_dim, output_lang.n_words, device ).to(device)
    # decoder = AttnDecoderRNN( h_dim, output_lang.n_words, MAX_LENGTH, device, dropout_p=0.1).to(device)

    # encoder, decoder = load_model('models/')

    save_path = PATH+attention_type+'-attention-rnn'+rnn_type+'-checkpt_'
    try:
        encoder, decoder, encoder_optimiser, decoder_optimiser = load_checkpoint(encoder, decoder, encoder_optimiser, decoder_optimiser, save_path )
    except:
        trainIters(encoder, decoder, encoder_optimiser, decoder_optimiser, 75000, input_lang, output_lang, device, save_path, print_every=100)

    evaluateRandomly(encoder, decoder, pairs, input_lang, output_lang, device, n=15)

    evaluateAndShowAttention("elle a cinq ans de moins que moi .", encoder, decoder, input_lang, output_lang, device)
    evaluateAndShowAttention("elle est trop petit .", encoder, decoder, input_lang, output_lang, device)
    evaluateAndShowAttention("je ne crains pas de mourir .", encoder, decoder, input_lang, output_lang, device)
    evaluateAndShowAttention("c est un jeune directeur plein de talent .", encoder, decoder, input_lang, output_lang, device)
