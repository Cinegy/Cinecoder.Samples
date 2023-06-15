/* Copyright 2022-2023 Cinegy GmbH.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/

using System.Runtime.InteropServices;

namespace SimpleBenchmark.SerializableModels.Settings
{
    public class AppConfig
    {
        private bool _liveConsole = false;

        public bool AsapMode { get; set; }

        public bool UseGpu { get; set; }

        public int VideoWidth { get; set; } = 1920;

        public int VideoHeight { get; set; } = 1080;
        
        public int OutputBitrate { get; set; } = 5000000;

        public int GopN { get; set; } = 15;

        public int GopM { get; set; } = 1;

        public string Ident { get; set; } = "Benchmark1";

        public string Label { get; set; }

        public bool LiveConsole
        {
            get => RuntimeInformation.IsOSPlatform(OSPlatform.Windows) && _liveConsole;
            set => _liveConsole = value;
        }

        public MetricsSetting Metrics { get; set; } = new();

    }
}
