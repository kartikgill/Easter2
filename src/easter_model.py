import config
import tensorflow
import tensorflow.keras.backend as K
from data_loader import data_loader

def ctc_loss(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(
        labels, 
        y_pred, 
        input_length, 
        label_length
    )

def ctc_custom(args):
    """
    custom CTC loss
    """
    y_pred, labels, input_length, label_length = args
    ctc_loss = K.ctc_batch_cost(
        labels, 
        y_pred, 
        input_length, 
        label_length
    )
    p = tensorflow.exp(-ctc_loss)
    gamma = 0.5
    alpha=0.25 
    return alpha*(K.pow((1-p),gamma))*ctc_loss

def batch_norm(inputs):
    return tensorflow.keras.layers.BatchNormalization(
        momentum= config.BATCH_NORM_DECAY, 
        epsilon = config.BATCH_NORM_EPSILON
    )(inputs)

def add_global_context(data, filters):
    """
    1D Squeeze and Excitation Layer. 
    """
    pool = tensorflow.keras.layers.GlobalAveragePooling1D()(data)
    
    pool = tensorflow.keras.layers.Dense(
        filters//8, 
        activation='relu'
    )(pool)
    
    pool = tensorflow.keras.layers.Dense(
        filters, 
        activation='sigmoid'
    )(pool) 
    
    final = tensorflow.keras.layers.Multiply()([data, pool])
    return final

def easter_unit(old, data, filters, kernel, stride, dropouts):
    """
    Easter unit with dense residual connections
    """
    old = tensorflow.keras.layers.Conv1D(
        filters = filters, 
        kernel_size = (1), 
        strides = (1),
        padding = "same"
    )(old)
    old = batch_norm(old)
    
    this = tensorflow.keras.layers.Conv1D(
        filters = filters, 
        kernel_size = (1), 
        strides = (1),
        padding = "same"
    )(data)
    this = batch_norm(this)
    
    old = tensorflow.keras.layers.Add()([old, this])
    
    #First Block
    data = tensorflow.keras.layers.Conv1D(
        filters = filters, 
        kernel_size = (kernel), 
        strides = (stride),
        padding = "same"
    )(data)
    
    data = batch_norm(data)
    data = tensorflow.keras.layers.Activation('relu')(data)
    data = tensorflow.keras.layers.Dropout(dropouts)(data)
    
    #Second Block
    data = tensorflow.keras.layers.Conv1D(
        filters = filters, 
        kernel_size = (kernel), 
        strides = (stride),
        padding = "same"
    )(data)
    
    data = batch_norm(data)
    data = tensorflow.keras.layers.Activation('relu')(data)
    data = tensorflow.keras.layers.Dropout(dropouts)(data)
    
    #Third Block
    data = tensorflow.keras.layers.Conv1D(
        filters = filters, 
        kernel_size = (kernel), 
        strides = (stride),
        padding = "same"
    )(data)
    
    data = batch_norm(data)
    
    #squeeze and excitation
    data = add_global_context(data, filters)
    
    final = tensorflow.keras.layers.Add()([old,data])
    
    data = tensorflow.keras.layers.Activation('relu')(final)
    data = tensorflow.keras.layers.Dropout(dropouts)(data)
       
    return data, old

def Easter2():
    input_data = tensorflow.keras.layers.Input(
        name='the_input', 
        shape = config.INPUT_SHAPE
    )
    
    data = tensorflow.keras.layers.Conv1D(
        filters = 128, 
        kernel_size = (3), 
        strides = (2), 
        padding = "same"
    )(input_data)
    
    data = batch_norm(data)
    data = tensorflow.keras.layers.Activation('relu')(data)
    data = tensorflow.keras.layers.Dropout(0.2)(data)

    data = tensorflow.keras.layers.Conv1D(
        filters = 128, 
        kernel_size = (3), 
        strides = (2), 
        padding = "same"
    )(data)
    
    data = batch_norm(data)
    data = tensorflow.keras.layers.Activation('relu')(data)
    data = tensorflow.keras.layers.Dropout(0.2)(data)

    old = data

    # 3 * 3 Easter Blocks (with dense residuals)
    data, old = easter_unit(old, data, 256, 5, 1, 0.2)
    data, old = easter_unit(old, data, 256, 7, 1, 0.2 )
    data, old = easter_unit(old, data, 256, 9, 1, 0.3 )

    data = tensorflow.keras.layers.Conv1D(
        filters = 512, 
        kernel_size = (11), 
        strides = (1), 
        padding = "same", 
        dilation_rate = 2
    )(data)
    
    data = batch_norm(data)
    data = tensorflow.keras.layers.Activation('relu')(data)
    data = tensorflow.keras.layers.Dropout(0.4)(data)

    data = tensorflow.keras.layers.Conv1D(
        filters = 512, 
        kernel_size = (1), 
        strides = (1), 
        padding = "same"
    )(data)
    
    data = batch_norm(data)
    data = tensorflow.keras.layers.Activation('relu')(data)
    data = tensorflow.keras.layers.Dropout(0.4)(data)

    data = tensorflow.keras.layers.Conv1D(
        filters = config.VOCAB_SIZE, 
        kernel_size = (1), 
        strides = (1), 
        padding = "same"
    )(data)
    
    y_pred = tensorflow.keras.layers.Activation('softmax',name="Final")(data)

    # print model summary
    tensorflow.keras.models.Model(inputs = input_data, outputs = y_pred).summary()
 
    # Defining other training parameters
    Optimizer = tensorflow.keras.optimizers.Adam(lr = config.LEARNING_RATE)
    
    labels = tensorflow.keras.layers.Input(
        name = 'the_labels', 
        shape=[config.OUTPUT_SHAPE], 
        dtype='float32'
    )
    input_length = tensorflow.keras.layers.Input(
        name='input_length', 
        shape=[1],
        dtype='int64'
    )
    label_length = tensorflow.keras.layers.Input(
        name='label_length',
        shape=[1],
        dtype='int64'
    )
    
    output = tensorflow.keras.layers.Lambda(
        ctc_custom, output_shape=(1,),name='ctc'
    )([y_pred, labels, input_length, label_length])

    # compiling model
    model = tensorflow.keras.models.Model(
        inputs = [input_data, labels, input_length, label_length], outputs= output
    )
    
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer = Optimizer)
    return model

def train():
    #Creating Easter2 object
    model = Easter2()
    
    # Loading checkpoint for transfer/resuming learning
    if config.LOAD:
        print ("Intializing from checkpoint : ", config.LOAD_CHECKPOINT_PATH)
        model.load_weights(config.LOAD_CHECKPOINT_PATH)
        print ("Init weights loaded successfully....")
        
    # Loading Metadata, about training, validation and Test sets
    print ("loading metdata...")
    training_data = data_loader(config.DATA_PATH, config.BATCH_SIZE)
    validation_data = data_loader(config.DATA_PATH, config.BATCH_SIZE)
    test_data = data_loader(config.DATA_PATH, config.BATCH_SIZE)

    training_data.trainSet()
    validation_data.validationSet()
    test_data.testSet()

    print("Training Samples : ", len(training_data.samples))
    print("Validation Samples : ", len(validation_data.samples))
    print("Test Samples : ", len(test_data.samples))
    print("CharList Size : ", len(training_data.charList))
    
    # callback arguments
    CHECKPOINT = tensorflow.keras.callbacks.ModelCheckpoint(
        filepath = config.CHECKPOINT_PATH,
        monitor='loss', 
        verbose=1, 
        mode='min', 
        period = 2
    )
    
    TENSOR_BOARD = tensorflow.keras.callbacks.TensorBoard(
        log_dir=config.LOGS_DIR, 
        histogram_freq=0, 
        write_graph=True,
        write_images=False, 
        embeddings_freq=0
    )
    
    # steps per epoch calculation based on number of samples and batch size
    STEPS_PER_EPOCH = len(training_data.samples)//config.BATCH_SIZE
    VALIDATION_STEPS = len(validation_data.samples)//config.BATCH_SIZE

    # Start training with given parameters
    print ("Training Model...")
    model.fit_generator(
        generator = training_data.getNext(), 
        steps_per_epoch = STEPS_PER_EPOCH,
        epochs = config.EPOCHS,
        callbacks=[CHECKPOINT, TENSOR_BOARD],
        validation_data = validation_data.getNext(), 
        validation_steps = VALIDATION_STEPS
    )