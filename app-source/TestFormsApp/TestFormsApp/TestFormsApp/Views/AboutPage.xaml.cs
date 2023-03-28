using System;
using System.Threading.Tasks;
using Xamarin.Essentials;
using Xamarin.Forms;

namespace TestFormsApp.Views
{
    public partial class AboutPage : ContentPage
    {
        public AboutPage()
        {
            InitializeComponent();
        }

        async void OnButtonClicked(object sender, EventArgs e)
        {
            // Launch the specified URL in the system browser.

            // get the platform and show it here
            if(Device.RuntimePlatform == Device.iOS)
            {
                // iOS
                platformButton.Text = "You are on iOS";
                await Navigation.PushAsync(new AboutPage());
            }
            else if (Device.RuntimePlatform == Device.Android)
            {
                // android
                platformButton.Text = "You are on Android.";
                await Navigation.PushAsync(new AboutPage());

                
            }
        }
    }
}