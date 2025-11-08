import os, sys
from functools import reduce
from pm_helper_classes import Dataset
from layer_visualizer import LayerVisualizer


class Commands:
    """Class containing command methods for the Shell.
     Methods that don't use global variables or decorators should go here
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
            self.message( "Model not found. Please set the model in context first" )
            return

        model = model_info.model

        # Add LLM analysis if available
        if hasattr(self, 'llm_service') and self.llm_service and self.llm_service.is_available():
            self.message( "\n" + "=" * 60 )
            self.message( "LLM Analysis:" )
            self.message( "=" * 60 )
            try:
                description = self.llm_service.describe_architecture(model, model_info.name)
                self.message( description )
                self.message( "=" * 60 )
            except Exception as e:
                self.error( f"LLM analysis failed: {e}" )
        else:
            self.message( "\nLLM service not available. Configure a provider first." )

    do_show_summary = do_summary


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
                self.message( "  {:<20} = {}".format( attr, value ) )

        if self.llm_service.is_available():
            provider = self.llm_service.get_provider()
            info = provider.get_model_info()
            self.message( f"LLM provider: {provider.get_provider_name()} Model: {info.get('model', 'unknown')}" )

    do_show_conf = do_show_config


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
