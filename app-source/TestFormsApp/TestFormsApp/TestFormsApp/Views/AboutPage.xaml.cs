using System;
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

        async void XamarinButtonClicked(object sender, EventArgs e)
        {
            // Launch the specified URL in the system browser.
            await Launcher.OpenAsync("https://aka.ms/xamarin-quickstart");
        }

        async void PlasticButtonClicked(object sender, EventArgs e)
        {
            await Launcher.OpenAsync("https://www.theplasticproject.org/");
        }
    }
}