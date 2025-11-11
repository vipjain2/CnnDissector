import os, sys
import functools
import numpy as np
from functools import reduce
from collections import OrderedDict
from pm_helper_classes import Dataset
from layer_visualizer import LayerVisualizer
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from PIL import Image
from torchvision import transforms


class Commands:
    """Class containing command methods for the Shell.
    """

    def do_quit( self, args ):
        """Exits the shell
        """
        self.message( "Exiting shell" )
        self.close()
        raise SystemExit


    def do_summary( self, args ):
        """Prints pytorch model summary with optional LLM analysis
        """
        model_info, _ = self.get_info_from_context( args )
        if model_info is None:
            return

        model = model_info.model

        # Add LLM analysis if available
        if hasattr(self, 'llm_service') and self.llm_service and self.llm_service.is_available():
            self.message( "\n" + "=" * 50 )
            self.message( "Analysis:" )
            self.message( "=" * 50 )
            try:
                description = self.llm_service.describe_architecture(model, model_info.name)
                self.message( description )
                self.message( "=" * 50 )
            except Exception as e:
                self.error( f"LLM analysis failed: {e}" )
        else:
            self.message( "\nLLM service not available. Configure a provider first." )

    do_show_summary = do_summary


    def do_show_config( self, args ):
        """Display the current configuration settings
        Usage: show config
        """
        self.message( "Configuration settings:" )
        config_attrs = [ attr for attr in dir( self.config ) if not attr.startswith( '_' ) ]

        if not config_attrs:
            self.message( "  No configuration attributes set" )
        else:
            for attr in sorted( config_attrs ):
                value = getattr( self.config, attr )
                self.message( f"  {attr:<20} = {value}" )

        if self.llm_service.is_available():
            provider = self.llm_service.get_provider()
            info = provider.get_model_info()
            self.message( "LLM provider:" )
            self.message( f"  {'Name':<10}: {provider.get_provider_name()}" )
            self.message( f"  {'Model':<10}: {info.get( 'model', 'unknown' )}" )

    do_show_conf = do_show_config

    ####################################
    # Decorators
    ####################################
    def supports_compare( func ):    # pylint: disable=no-self-argument
        """We store self.compare on the stack to make the decorator reentrant
        This avoids recursion in case a function that supports this decorator
        is called again inside the decorator
        """

        @functools.wraps( func )
        def wrapper( self, *kargs, **kwargs ):
            compare = self.compare
            self.stack.append( self.compare )
            self.compare = None

            if compare:
                self.fig.set_mode( "dual" )
    
            try:
                if compare == "image":
                    self.do_show_image( *kargs, **kwargs )
                elif compare is not None:
                    func( self, compare )       #pylint: disable=not-callable

                func( self, *kargs, **kwargs )  #pylint: disable=not-callable
            except:
                raise
            finally:
                self.compare = self.stack.pop()
                if compare:
                    self.fig.set_mode( "single" )

        return wrapper

    #############################################
    # All the image related functions go here
    #############################################
    def do_load_image( self, args ):
        """Load a single image from the path specified
        Usage: load image [ path ]
        """
        image_path = os.path.join( self.config.image_path, args )
        if not os.path.isfile( image_path ):
            self.error( "Image not found")
            return
        self.message( "Loading image {}".format( image_path ) )
        image = Image.open( image_path )
        transform = transforms.Compose( [ transforms.Resize( ( self.image_size, self.image_size ) ),
                                         
                                          transforms.ToTensor() ] )
        image = transform( image ).float().unsqueeze( 0 )
        # Store image in current frame's namespace
        self.cur_frame.f_locals[ 'image' ] = image
        self.cur_frame.f_globals[ 'image'] = image

    def do_show_image( self, args ):
        """Display an image array:
        Usage: show image [ image_var ]

        If an (optional) image_var is specified, it's considered as a global
        variable referencing an image array and tries to show this image.
        If no args are specified, show the image referenced by the
        global "image" variable.
        """
        if not args:
            args = "image"
        img = self.in_place_eval( args )
        if img is None:
            self.error( "Could not display image" )
            return

        if not self.fig.imshow( img ):
            self.error( "Unsupported image type" )
            return

    do_show_img = do_show_image


    def do_image_next( self, args ):
        """Load the next available image from a dataset:
        Usage: image next
        
        This command operates on a dataset. A dataset must be configued for this 
        command. If there are no more images available to be loaded, it keeps the 
        last available image.
        The "image" global variable points to the loaded image.
        """ 
        if self.dataset is None:
            self.message( "Please configure a dataset first" )
            return

        self.dataset.next()
        image = self.dataset.load()
        # Store image in current frame's namespace
        self.cur_frame.f_locals[ 'image' ] = image
        self.cur_frame.f_globals[ 'image'] = image
        
        self.fig.imshow( image )


    def do_heatmap_next( self, args ):
        global image
        
        if self.dataset is None:
            self.error( "No dataset configured" )
            return

        self.dataset.next()
        image = self.dataset.load()
        # Store image in current frame's namespace
        self.cur_frame.f_locals[ 'image' ] = image
        self.cur_frame.f_globals[ 'image'] = image
        
        self.do_show_heatmap( args=args )

    do_heat_next = do_heatmap_next


    def do_activations_next( self, args ):
        global image
        
        if self.dataset is None:
            self.error( "No dataset configured" )
            return

        self.dataset.next()
        image = self.dataset.load()
        # Store image in current frame's namespace
        self.cur_frame.f_locals[ 'image' ] = image
        self.cur_frame.f_globals[ 'image'] = image
        
        self.do_show_activations( args=args )

    do_act_next = do_activations_next


    @supports_compare
    def do_show_first_layer_weights( self, args ):
        """Display the weight vectors in a layer as a grid:
        Usage: show first layer weights [ model name/tensor ]
        
        If no arguments are given, it displays the first Conv layer weights in the default model
        set in the context
        If an optional model name if provided, it displays the first layer weight for that
        model
        Optionally, a weight tensor can be given as an argument.
        """
        if not args and not self.cur_model:
            self.error( "No default model is set. Please set a model in context first." )
            return
        
        title = "Tensor {}".format( args )
        _weight = None
        if args and args not in self.models:
            _weight = self.in_place_eval( args )
            if _weight is None or not isinstance( _weight, torch.Tensor ):
                self.error( "Can not display \"{}\". Not a model in context, or a tensor.".format( args ) )
                return

            if _weight.dim() != 4 and _weight.size[ 1 ] not in ( 3, 1 ):
                self.error( "Tensor \"{}\" is incorrect shape for display".format( args ) )
                return

        if _weight is None: 
            model_info = self.models[ args ] if args else self.cur_model
            title = "{} first layer weights".format( model_info.name )
            _, conv = model_info.find_first_instance( type=nn.Conv2d )
            if not conv:
                self.error( "No Conv2d layer found" )
                return
            _weight = conv.weight.detach()
        
        self.show_weights_as_grid( _weight, title )

    do_show_flw = do_show_first_layer_weights


    @supports_compare
    def do_show_activations( self, args ):
        model_info, layer_info = self.get_info_from_context( args )

        if model_info is None:
            return

        img = self.load_from_global( "image" )
        if img is None:
            self.error( "Please load an input image first" )
            return

        if self.data_post_process_fn:
            self.message( "Using processing function {}".format( self.data_post_process_fn.__name__ ) )

        layer_info.register_forward_hook()
        net = model_info.model
        out = net( image )
        self.message( "{} out: {}".format( model_info.name, out.argmax() ) )

        title = "{}{} activations".format( model_info.name, layer_info.id )
        self.display_bargraph( layer_info.data(), title, reduce_fn=self.data_post_process_fn )

    do_show_act = do_show_activations


    @supports_compare
    def do_show_heatmap( self, args ):
        model_info, layer_info = self.get_info_from_context( args )
        
        if model_info is None:
            self.message( "Please set a model in context first" )
            return

        img = self.load_from_global( "image" )
        if img is None:
            self.error( "No input image available" )
            return

        id, layer = model_info.find_last_instance( layer=nn.Conv2d )
        layer_info = model_info.get_layer_info( id, layer )
        layer_info.register_forward_hook()

        net = model_info.model
        idx = net( image ).argmax()
        
        _, fc = model_info.find_last_instance( layer=nn.Linear )
        fc_weights = fc.weight[ idx ].data.numpy()

        activations = layer_info.data()[ 0 ].data.numpy()        
        nc, h, w = activations.shape
        
        cam = fc_weights.reshape( 1, nc ).dot( activations.reshape( nc, h * w ) )
        cam = cam.reshape( h, w )
        cam = ( cam - np.min( cam ) ) / np.max( cam )
        cam = Image.fromarray( cam )

        _, _, h, w = img.size()     
        cam = cam.resize( ( h, w ), Image.BICUBIC )
        cam = transforms.ToTensor()( cam )[ 0 ]
        
        msg = "{} guess: {}".format( model_info.name, idx )
        self.message( msg )
        window = self.fig.get_or_create_window()
        window.add_title( msg )
        window.add_image( img )
        window.add_image( cam, cmap="jet", alpha=0.5 )
        window.show()

    do_show_heat = do_show_heatmap


    @supports_compare
    def do_show_weights( self, args ):
        model_info, layer_info = self.get_info_from_context( args )
        if model_info is None:
            return

        id, layer = layer_info.id, layer_info.layer
        self.message( "Current layer is {}: {}".format( id, layer ) )

        if self.data_post_process_fn:
            self.message( "Post processing function is {}".format( self.data_post_process_fn.__name__ ) )

        try:
            data = layer_info.layer.weight.unsqueeze( 0 )
        except:
            self.error( "Current layer has no weights")
        else:
            title = "{} weights".format( layer_info.id )
            self.display_bargraph( data, title, reduce_fn=self.data_post_process_fn )

    do_show_weight = do_show_weights
    do_show_wei = do_show_weights


    @supports_compare
    def do_show_grads( self, args ):
        model_info, layer_info = self.get_info_from_context( args )
        if model_info is None:
            return
    
        id, layer = layer_info.id, layer_info.layer
        self.message( "Current layer is {}: {}".format( id, layer ) )

        if self.data_post_process_fn:
            self.message( "Post processing function is {}".format( self.data_post_process_fn.__name__ ) )

        try:
            data = layer_info.layer.weight.grad.unsqueeze( 0 )
        except:
            self.error( "Current layer has no gradients" )
        else:
            title = "{} gradients".format( layer_info.id )
            self.display_bargraph( data, title, reduce_fn=self.data_post_process_fn )


    #############################################
    # All the 'model' related functions go here
    #############################################
    def do_set_context( self, args ):
        model_name = args if args else "model"
        model = self.load_from_global( model_name )
        if model is None:
            self.error( "Could not find a model by name \"{}\"".format( model_name ) )
            return

        if not isinstance( model, nn.Module ):
            self.error( "{} is not a valid model" )
            return

        self.set_model( model_name, model )

        self.message( "Context now is \"{}\"".format( model_name ) )
        self.fig.set_window_title( model_name )
    
    do_set_ctx = do_set_context


    def do_resync( self, args ):
        if not args:
            self.error( "Please provide a model name" )
            return

        if args not in self.models:
            self.error( "Model \"{}\" not in context".format( args ) )
            return

        self.resync_model( args )


    def do_load_checkpoint( self, args ):
        """Load a checkpoint file into the model:
        Usage: load checkpoint [ filename ]

        If no file is specified, checkpoint_name specified in the
        config file is used"""
        model_info = self.cur_model
        if model_info is None:
            self.error( "No default model set in context." )
            return
        
        model = model_info.model

        if args:
            file = os.path.join( self.config.checkpoint_path, args )
        else:
            file = os.path.join( self.config.checkpoint_path, self.config.checkpoint_name )
        if not os.path.isfile( file ):
            self.error( "Checkpoint file not found" )
            return

        chkpoint = torch.load( file, map_location="cpu" )
        self.message( "Model \"{}\", loading checkpoint: {}".format( model_info.name, file ) )
        
        state_dict = chkpoint[ "model" ]

        try:
            model.load_state_dict( state_dict )
        except RuntimeError:
            new_state_dict = OrderedDict( [ ( k[ 7: ], v ) for k, v in state_dict.items() 
                                                            if k.startswith( "module" ) ] )
            model.load_state_dict( new_state_dict )

    do_load_chkp = do_load_checkpoint
    do_laod_chkp = do_load_checkpoint


    def do_show_layers( self, args ):
        """Prints pytorch layers list
        """
        model_info, _ = self.get_info_from_context( args )
        if model_info is None:
            self.message( "Model not found. Please set the model in context first" )
            return

        model = model_info.model
        try:
            self.message( "Model: {}".format( model_info.name ) )
            self.message( "-" * 60 )
            for name, module in model.named_modules():
                if name:  # Skip the root module
                    self.message( "{:<40} {}".format( name, module.__class__.__name__ ) )
        except:
            self.error( sys.exc_info()[ 1 ] )


    def do_nparams( self, args ):
        """Print the total number of parameters in a model
        """
        model_info, _  = self.get_info_from_context( args )
        if model_info is None:
            self.message( "Model \"{}\" not found. Please set the model in context first".format( args ) )
            return

        model = model_info.model
        n =  sum( reduce( lambda x, y: x * y, p.size() ) for p in model.parameters())
        print( "{:,}".format( n ) )

    do_nparam = do_nparams
    do_show_nparams = do_nparams

    @supports_compare
    def do_infer_image( self, args ):
        """Run inference on image:
        Usage: infer image [ model_name ]

        If an optional "model_name" is provided, inference is run
        on that model. The "model_name" must be present in context.
        If no "model_name" is provided, inference is run on the current 
        model set in the context.

        Input image is taken from the global "image" variable.
        """
        model_info, _ = self.get_info_from_context( args )
        if model_info is None:
            return

        img = self.load_from_global( "image" )
        if img is None:
            self.error( "Please load an input image first" )
            return

        net = model_info.model
        out = net( image )
        probs, idxs = softmax( out, dim=1 ).topk( 5, dim=1 )
        self.message( "{}:".format( model_info.name ) )
        for idx, prob in zip( idxs[ 0 ], probs[ 0 ] ):
            self.message( "{:<10}{:4.2f}".format( idx.data, prob.data * 100 ) )

    do_infer = do_infer_image
    do_show_infer = do_infer_image


    def do_visualizer( self, args ):
        if self.cur_model is None:
            self.error( "No current model set in context" )
            return
        image = self.load_from_global( "image" )
        if image is None:
            self.error( "Please load an input image first" )
            return

        visual = LayerVisualizer( self.cur_model )
        visual.set_image( image )
        visual.start()


    def do_up( self, args ):
        if not self.cur_model:
            self.error( "Please load a model first" )
            return

        if not self.cur_model.up():
            self.message( "Already at top" )
        id, layer = self.cur_model.get_cur_id_layer()
        self.message( "Current layer is {}: {}".format( id, layer ) )


    def do_down( self, args ):
        if not self.cur_model:
            self.error( "Please load a model first" )
            return
        if not self.cur_model.down():
            self.message( "Already at bottom" )
        id, layer = self.cur_model.get_cur_id_layer()
        self.message( "Current layer is {}: {}".format( id, layer ) )


    def do_set_class( self, args ):
        if self.dataset is None:
            self.error( "No dataset is configured" )
        idx = int( args ) if args else 0
        if not self.dataset.set_class( idx ):
            self.error( "Could not set class to {}".format( args ) )


    def do_set_compare( self, args ):
        if not args:
            self.compare = None
        elif args not in self.models and args not in ( "image", "flw" ):
            self.error( "Model \"{}\" not in context".format( args ) )
            return
        else:
            self.compare = args

    do_set_comp = do_set_compare


    def do_set_dataset( self, path ):
        expanded_path = self.expand_path( path )
        if not os.path.exists( expanded_path ):
            self.error( "Dataset path does not exist: {}".format( expanded_path ) )
            return
        self.dataset = Dataset( expanded_path )

    do_set_dataset_path = do_set_dataset
    do_set_data_path = do_set_dataset


    def do_dataset_suffix( self, args ):
        if self.dataset.suffix( args ):
            self.message( "Current dataset is: {}".format( self.dataset.data_path ) )
        else:
            self.error( "Could not change suffix" )


    def do_set_post_process( self, args ):
        if args == "relu":
            fn = torch.nn.ReLU()
        elif args == "mean":
            fn = torch.mean     # pylint: disable=no-member
        elif args == "max":
            fn = torch.max      # pylint: disable=no-member
        elif args == "none" or args == "None":
            fn = None
        else:
            fn = self.load_from_global( args )
            if not fn:
                self.error( "Could not find function \"{}\"".format( args ) )
                return

        if not fn:
            self.message( "Removing post processing function" )
            self.data_post_process_fn = None
            return

        if fn and not callable( fn ):
            self.error( "Not a valid function" )
            return

        self.data_post_process_fn = fn
        if not hasattr( self.data_post_process_fn, "__name__" ):
            self.data_post_process_fn.__name__ = args
        self.message( "Post process function is {}".format( self.data_post_process_fn.__name__ ) )

    do_set_postp = do_set_post_process


    def do_assign( self, args ):
        args = args.split()

        def _set_name_and_check_in_use( default ):
            var = args[ 1 ] if len( args ) > 1 else default
            if var in self.cur_frame.f_globals or var in self.cur_frame.f_locals:
                self.error( "Variable \"{}\" already in use.".format( var ) )
                self.help( "please \"del {}\" first, if you want to reassign this var".format( var ) )
                return
            else:
                return var

        if not self.cur_model:
            self.error( "Please set a model in context first")
            return
        if not args:
            self.error( "No valid options provided" )
            return

        if args[ 0 ] == "layer":
            var = _set_name_and_check_in_use( "layer" )
            self.cur_frame.f_globals[ var ] = self.cur_model.cur_layer.layer
            self.cur_frame.f_locals[ var ] = self.cur_model.cur_layer.layer

        elif args[ 0 ] == "weight":
            var = _set_name_and_check_in_use( "weight" )
            self.cur_frame.f_globals[ var ] = self.cur_model.cur_layer.layer.weight.data.detach().clone()
            self.cur_frame.f_locals[ var ] = self.cur_model.cur_layer.layer.weight.data.detach().clone()

        elif args[ 0 ] == "out" or args[ 0 ] == "outsq":
            var = _set_name_and_check_in_use( "out" )
            self.cur_model.cur_layer.register_forward_hook()

            # Get image from current frame namespace
            image = self.cur_frame.f_locals.get('image') or self.cur_frame.f_globals.get('image')

            if image is None:
                self.cur_frame.f_globals[ var ] = None
                self.cur_frame.f_locals[ var ] = None
            else:
                _ = self.cur_model.model( image )
            if args[ 0 ] == "outsq":
                data = self.cur_model.cur_layer.data()
                result = data.squeeze( 0 ) if data is not None else None
                self.cur_frame.f_globals[ var ] = result
                self.cur_frame.f_locals[ var ] = result
            else:
                result = self.cur_model.cur_layer.data()
                self.cur_frame.f_globals[ var ] = result
                self.cur_frame.f_locals[ var ] = result
        else:
            self.error( "Invalid comand option \"{}\"".format( args[ 0 ] ) )
