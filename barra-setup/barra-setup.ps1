# ============================================================================
# barra-setup.ps1 — Die GUI. Ein Fenster, drei Seiten: Einrichten → Flashen → Fertig.
# Windows-nativ (adb/fastboot aus .\tools), kein WSL/Admin. Kern: barra-core.ps1 im Runspace.
# UI-Texte aus .\i18n\<lang>.properties (barra-i18n.ps1, T 'key' args...); Sprach-Dropdown auf Seite 1.
# Start: barra-setup.vbs (ohne Konsole) oder powershell -File barra-setup.ps1
# ============================================================================
Add-Type -AssemblyName PresentationFramework, PresentationCore, WindowsBase, System.Web
$ErrorActionPreference = 'Continue'
$Kit = Split-Path -Parent $MyInvocation.MyCommand.Path
$CfgFile = Join-Path $env:LOCALAPPDATA 'barra\setup.json'
. (Join-Path $Kit 'barra-i18n.ps1')

function Load-Cfg { if (Test-Path $CfgFile) { try { return (Get-Content $CfgFile -Raw | ConvertFrom-Json) } catch {} }; $null }
function Save-Cfg($h){ New-Item -ItemType Directory -Force (Split-Path $CfgFile) | Out-Null; ($h | ConvertTo-Json) | Set-Content $CfgFile -Encoding UTF8 }
$saved = Load-Cfg

# Sprache: gespeichert -> Windows-UI-Sprache -> en
$script:lang = if ($saved -and $saved.lang) { $saved.lang } else {
  $c = [System.Globalization.CultureInfo]::CurrentUICulture.TwoLetterISOLanguageName
  if ((Get-BarraI18nLanguages (Join-Path $Kit 'i18n')) -contains $c) { $c } else { 'en' }
}
Initialize-BarraI18n -Dir (Join-Path $Kit 'i18n') -Lang $script:lang

[xml]$xaml = @"
<Window xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation" xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
        Title="barra Setup" Width="1060" Height="720" MinWidth="900" MinHeight="660" WindowStartupLocation="CenterScreen"
        Background="#0A0C10" FontFamily="Segoe UI" FontSize="14">
  <Window.Resources>
    <Style TargetType="Button">
      <Setter Property="Background" Value="#5B8DEF"/><Setter Property="Foreground" Value="White"/><Setter Property="BorderThickness" Value="0"/>
      <Setter Property="Padding" Value="20,9"/><Setter Property="FontSize" Value="14"/><Setter Property="Cursor" Value="Hand"/>
      <Setter Property="Template"><Setter.Value><ControlTemplate TargetType="Button">
        <Border x:Name="b" Background="{TemplateBinding Background}" CornerRadius="8" Padding="{TemplateBinding Padding}"><ContentPresenter HorizontalAlignment="Center" VerticalAlignment="Center"/></Border>
        <ControlTemplate.Triggers><Trigger Property="IsEnabled" Value="False"><Setter TargetName="b" Property="Opacity" Value="0.4"/></Trigger></ControlTemplate.Triggers>
      </ControlTemplate></Setter.Value></Setter>
    </Style>
    <Style TargetType="TextBox"><Setter Property="Background" Value="#0A0C10"/><Setter Property="Foreground" Value="#F0F2F6"/><Setter Property="BorderBrush" Value="#2E323C"/><Setter Property="Padding" Value="8,6"/><Setter Property="FontSize" Value="14"/><Setter Property="CaretBrush" Value="#F0F2F6"/></Style>
    <Style TargetType="PasswordBox"><Setter Property="Background" Value="#0A0C10"/><Setter Property="Foreground" Value="#F0F2F6"/><Setter Property="BorderBrush" Value="#2E323C"/><Setter Property="Padding" Value="8,6"/><Setter Property="FontSize" Value="14"/><Setter Property="CaretBrush" Value="#F0F2F6"/></Style>
    <Style TargetType="ComboBox"><Setter Property="Background" Value="#0A0C10"/><Setter Property="Foreground" Value="#1B1F27"/><Setter Property="FontSize" Value="13"/></Style>
    <Style TargetType="TextBlock"><Setter Property="Foreground" Value="#C9CFD8"/></Style>
    <Style x:Key="Lbl" TargetType="TextBlock"><Setter Property="Foreground" Value="#8B939F"/><Setter Property="FontSize" Value="12"/><Setter Property="Margin" Value="0,10,0,4"/></Style>
    <Style x:Key="Grp" TargetType="TextBlock"><Setter Property="Foreground" Value="#5B8DEF"/><Setter Property="FontSize" Value="12"/><Setter Property="FontWeight" Value="SemiBold"/><Setter Property="Margin" Value="0,2,0,2"/></Style>
    <Style x:Key="Hint" TargetType="TextBlock"><Setter Property="Foreground" Value="#606772"/><Setter Property="FontSize" Value="12"/><Setter Property="Margin" Value="0,4,0,0"/><Setter Property="TextWrapping" Value="Wrap"/></Style>
  </Window.Resources>
  <Grid Margin="20">
    <Grid.RowDefinitions><RowDefinition Height="Auto"/><RowDefinition Height="*"/></Grid.RowDefinitions>
    <DockPanel Grid.Row="0" Margin="0,0,0,14">
      <Image x:Name="Logo" Height="44" Margin="0,0,12,0" VerticalAlignment="Bottom"/>
      <TextBlock Text="barra" FontSize="30" FontWeight="SemiBold" Foreground="#F0F2F6"/>
      <TextBlock x:Name="Sub" Text="" FontSize="16" Foreground="#8B939F" VerticalAlignment="Bottom" Margin="0,0,0,5"/>
      <ComboBox x:Name="FLang" DockPanel.Dock="Right" Width="110" VerticalAlignment="Bottom" Margin="12,0,0,3"/>
      <TextBlock x:Name="Pager" Text="" Foreground="#8B939F" HorizontalAlignment="Right" VerticalAlignment="Bottom" Margin="0,0,0,5"/>
    </DockPanel>

    <!-- SEITE 1: Einrichten -->
    <Border Grid.Row="1" x:Name="Page1" Background="#14181F" CornerRadius="14" Padding="26">
      <Grid>
        <Grid.RowDefinitions><RowDefinition Height="Auto"/><RowDefinition Height="*"/><RowDefinition Height="Auto"/></Grid.RowDefinitions>
        <StackPanel Grid.Row="0"><TextBlock x:Name="P1Title" Text="" FontSize="22" FontWeight="SemiBold" Foreground="#F0F2F6"/><TextBlock x:Name="P1Intro" Text="" Margin="0,4,0,0" TextWrapping="Wrap"/></StackPanel>
        <Grid Grid.Row="1" Margin="0,8,0,0">
          <Grid.ColumnDefinitions><ColumnDefinition Width="*"/><ColumnDefinition Width="22"/><ColumnDefinition Width="*"/><ColumnDefinition Width="22"/><ColumnDefinition Width="*"/></Grid.ColumnDefinitions>
          <StackPanel Grid.Column="0">
            <TextBlock x:Name="GrpAccess" Style="{StaticResource Grp}"/>
            <TextBlock x:Name="LUser" Style="{StaticResource Lbl}"/><TextBox x:Name="FUser"/>
            <TextBlock x:Name="LPass" Style="{StaticResource Lbl}"/><PasswordBox x:Name="FPass"/>
            <TextBlock x:Name="LPass2" Style="{StaticResource Lbl}"/><PasswordBox x:Name="FPass2"/>
            <TextBlock x:Name="LKey" Style="{StaticResource Lbl}"/>
            <TextBox x:Name="FKey" Height="96" TextWrapping="Wrap" AcceptsReturn="True" VerticalScrollBarVisibility="Auto" FontSize="12"/>
            <TextBlock x:Name="FKeyHint" Style="{StaticResource Hint}"/>
          </StackPanel>
          <StackPanel Grid.Column="2">
            <TextBlock x:Name="GrpNet" Style="{StaticResource Grp}"/>
            <TextBlock x:Name="LSsid" Style="{StaticResource Lbl}"/>
            <DockPanel><Button x:Name="BtnWifi" Content="" DockPanel.Dock="Right" Padding="12,6" Margin="8,0,0,0" Background="#2A2F3A"/><TextBox x:Name="FSsid"/></DockPanel>
            <TextBlock x:Name="SsidHint" Style="{StaticResource Hint}"/>
            <TextBlock x:Name="LPsk" Style="{StaticResource Lbl}"/><PasswordBox x:Name="FPsk"/>
            <TextBlock x:Name="LHost" Style="{StaticResource Lbl}"/><TextBox x:Name="FHost"/>
            <TextBlock x:Name="HostHint" Style="{StaticResource Hint}"/>
          </StackPanel>
          <StackPanel Grid.Column="4">
            <TextBlock x:Name="GrpOp" Style="{StaticResource Grp}"/>
            <TextBlock x:Name="LTz" Style="{StaticResource Lbl}"/><TextBox x:Name="FTz"/>
            <TextBlock x:Name="LChg" Style="{StaticResource Lbl}"/>
            <DockPanel LastChildFill="False"><TextBox x:Name="FChgS" Width="70" Text="50"/><TextBlock x:Name="LTo" Text="" VerticalAlignment="Center"/><TextBox x:Name="FChgE" Width="70" Text="55"/></DockPanel>
            <TextBlock x:Name="LChgHint" Style="{StaticResource Hint}"/>
            <TextBlock x:Name="LDisp" Style="{StaticResource Lbl}"/><TextBox x:Name="FDisp" Width="90" HorizontalAlignment="Left" Text="20"/>
            <TextBlock x:Name="DispHint" Style="{StaticResource Hint}"/>
          </StackPanel>
        </Grid>
        <DockPanel Grid.Row="2" Margin="0,16,0,0"><Button x:Name="BtnUnflash" Content="" DockPanel.Dock="Left" Background="#2A2F3A" Foreground="#C9CFD8" Padding="12,6"/><Button x:Name="BtnNext1" Content="" DockPanel.Dock="Right" HorizontalAlignment="Right"/><TextBlock x:Name="FErr" Foreground="#FF453A" VerticalAlignment="Center" Margin="14,0,0,0"/></DockPanel>
      </Grid>
    </Border>

    <!-- SEITE 1b: Pakete (eigener Bildschirm mit Erklaerungen) -->
    <Border Grid.Row="1" x:Name="PagePkg" Background="#14181F" CornerRadius="14" Padding="26" Visibility="Collapsed">
      <Grid>
        <Grid.RowDefinitions><RowDefinition Height="Auto"/><RowDefinition Height="*"/><RowDefinition Height="Auto"/></Grid.RowDefinitions>
        <StackPanel Grid.Row="0"><TextBlock x:Name="PkTitle" Text="" FontSize="22" FontWeight="SemiBold" Foreground="#F0F2F6"/><TextBlock x:Name="PkIntro" Text="" Margin="0,4,0,0" TextWrapping="Wrap"/></StackPanel>
        <ScrollViewer Grid.Row="1" x:Name="SvPkg" VerticalScrollBarVisibility="Auto" Margin="0,12,0,0">
          <StackPanel>
            <Border x:Name="CardLlm" Background="#0E1116" BorderBrush="#262A33" BorderThickness="1" CornerRadius="12" Padding="18,14" Margin="0,0,0,12">
              <StackPanel>
                <CheckBox x:Name="ChkLlm" Foreground="#F0F2F6" FontSize="15" FontWeight="SemiBold"/>
                <TextBlock x:Name="DescLlm" Margin="26,6,0,0" TextWrapping="Wrap" FontSize="13"/>
                <TextBlock x:Name="FactLlm" Style="{StaticResource Hint}" Margin="26,6,0,0"/>
              </StackPanel>
            </Border>
            <Border x:Name="CardStt" Background="#0E1116" BorderBrush="#262A33" BorderThickness="1" CornerRadius="12" Padding="18,14" Margin="0,0,0,12">
              <StackPanel>
                <CheckBox x:Name="ChkStt" Foreground="#F0F2F6" FontSize="15" FontWeight="SemiBold"/>
                <TextBlock x:Name="DescStt" Margin="26,6,0,0" TextWrapping="Wrap" FontSize="13"/>
                <TextBlock x:Name="FactStt" Style="{StaticResource Hint}" Margin="26,6,0,0"/>
              </StackPanel>
            </Border>
            <Border x:Name="CardPya" Background="#0E1116" BorderBrush="#262A33" BorderThickness="1" CornerRadius="12" Padding="18,14" Margin="0,0,0,12">
              <StackPanel>
                <CheckBox x:Name="ChkPya" Foreground="#F0F2F6" FontSize="15" FontWeight="SemiBold"/>
                <TextBlock x:Name="DescPya" Margin="26,6,0,0" TextWrapping="Wrap" FontSize="13"/>
                <TextBlock x:Name="FactPya" Style="{StaticResource Hint}" Margin="26,6,0,0"/>
              </StackPanel>
            </Border>
            <Border x:Name="CardTts" Background="#0E1116" BorderBrush="#262A33" BorderThickness="1" CornerRadius="12" Padding="18,14" Margin="0,0,0,12">
              <StackPanel>
                <CheckBox x:Name="ChkTts" Foreground="#F0F2F6" FontSize="15" FontWeight="SemiBold"/>
                <TextBlock x:Name="DescTts" Margin="26,6,0,0" TextWrapping="Wrap" FontSize="13"/>
                <TextBlock x:Name="FactTts" Style="{StaticResource Hint}" Margin="26,6,0,0"/>
              </StackPanel>
            </Border>
            <Border x:Name="CardWake" Background="#0E1116" BorderBrush="#262A33" BorderThickness="1" CornerRadius="12" Padding="18,14" Margin="0,0,0,12">
              <StackPanel>
                <CheckBox x:Name="ChkWake" Foreground="#F0F2F6" FontSize="15" FontWeight="SemiBold"/>
                <TextBlock x:Name="DescWake" Margin="26,6,0,0" TextWrapping="Wrap" FontSize="13"/>
                <TextBlock x:Name="FactWake" Style="{StaticResource Hint}" Margin="26,6,0,0"/>
              </StackPanel>
            </Border>
            <Border x:Name="CardImg" Background="#0E1116" BorderBrush="#262A33" BorderThickness="1" CornerRadius="12" Padding="18,14" Margin="0,0,0,12">
              <StackPanel>
                <CheckBox x:Name="ChkImg" Foreground="#F0F2F6" FontSize="15" FontWeight="SemiBold"/>
                <TextBlock x:Name="DescImg" Margin="26,6,0,0" TextWrapping="Wrap" FontSize="13"/>
                <TextBlock x:Name="FactImg" Style="{StaticResource Hint}" Margin="26,6,0,0"/>
              </StackPanel>
            </Border>
            <TextBlock x:Name="PkCoexWarn" Foreground="#FFB020" TextWrapping="Wrap" FontSize="13" Margin="0,2,0,0" Visibility="Collapsed"/>
          </StackPanel>
        </ScrollViewer>
        <DockPanel Grid.Row="2" Margin="0,16,0,0"><Button x:Name="BtnPkgBack" Content="" DockPanel.Dock="Left" Background="#2A2F3A" Foreground="#C9CFD8"/><Button x:Name="BtnPkgNext" Content="" DockPanel.Dock="Right" HorizontalAlignment="Right"/></DockPanel>
      </Grid>
    </Border>

    <!-- SEITE 1c: Modellwahl je Paket -->
    <Border Grid.Row="1" x:Name="PageModel" Background="#14181F" CornerRadius="14" Padding="26" Visibility="Collapsed">
      <Grid>
        <Grid.RowDefinitions><RowDefinition Height="Auto"/><RowDefinition Height="*"/><RowDefinition Height="Auto"/></Grid.RowDefinitions>
        <StackPanel Grid.Row="0"><TextBlock x:Name="MTitle" Text="" FontSize="22" FontWeight="SemiBold" Foreground="#F0F2F6"/><TextBlock x:Name="MIntro" Text="" Margin="0,4,0,0" TextWrapping="Wrap"/></StackPanel>
        <ScrollViewer Grid.Row="1" VerticalScrollBarVisibility="Auto" Margin="0,12,0,0"><StackPanel x:Name="MList"/></ScrollViewer>
        <DockPanel Grid.Row="2" Margin="0,16,0,0"><Button x:Name="BtnMBack" Content="" DockPanel.Dock="Left" Background="#2A2F3A" Foreground="#C9CFD8"/><Button x:Name="BtnMNext" Content="" DockPanel.Dock="Right" HorizontalAlignment="Right"/></DockPanel>
      </Grid>
    </Border>

    <!-- SEITE 2: Flashen -->
    <Grid Grid.Row="1" x:Name="Page2" Visibility="Collapsed">
      <Grid.ColumnDefinitions><ColumnDefinition Width="260"/><ColumnDefinition Width="*"/></Grid.ColumnDefinitions>
      <Grid.RowDefinitions><RowDefinition Height="*"/><RowDefinition Height="Auto"/></Grid.RowDefinitions>
      <Border Grid.Column="0" Background="#14181F" CornerRadius="14" Padding="16" Margin="0,0,14,0"><StackPanel x:Name="Steps"/></Border>
      <Border Grid.Column="1" Background="#14181F" CornerRadius="14" Padding="24">
        <Grid><Grid.RowDefinitions><RowDefinition Height="Auto"/><RowDefinition Height="Auto"/><RowDefinition Height="*"/><RowDefinition Height="Auto"/></Grid.RowDefinitions>
          <TextBlock x:Name="StepTitle" Grid.Row="0" Text="" FontSize="22" FontWeight="SemiBold" Foreground="#F0F2F6"/>
          <Border Grid.Row="1" x:Name="StateBox" Background="#0E1116" BorderBrush="#262A33" BorderThickness="1" CornerRadius="12" Padding="16,12" Margin="0,12,0,0">
            <StackPanel>
              <DockPanel>
                <TextBlock x:Name="StateDot" Text="●" FontSize="11" Foreground="#5B8DEF" VerticalAlignment="Center" Margin="0,0,8,0"/>
                <TextBlock x:Name="StateText" Text="" FontSize="12" Foreground="#8B939F" VerticalAlignment="Center"/>
              </DockPanel>
              <TextBlock x:Name="ProgText" Text="" FontSize="14.5" Foreground="#F0F2F6" Margin="0,7,0,0" TextWrapping="Wrap"/>
              <ProgressBar x:Name="Prog" Height="8" Margin="0,10,0,0" Minimum="0" Maximum="100" Value="0" Visibility="Collapsed">
                <ProgressBar.Template>
                  <ControlTemplate TargetType="ProgressBar">
                    <Grid>
                      <Border CornerRadius="4" Background="#262A33"/>
                      <Border x:Name="Pulse" CornerRadius="4" Background="#5B8DEF" Opacity="0.5" Visibility="Collapsed"/>
                      <Border x:Name="PART_Indicator" CornerRadius="4" Background="#5B8DEF" HorizontalAlignment="Left"/>
                      <Border x:Name="PART_Track" Background="Transparent"/>
                    </Grid>
                    <ControlTemplate.Triggers>
                      <Trigger Property="IsIndeterminate" Value="True">
                        <Setter TargetName="PART_Indicator" Property="Visibility" Value="Collapsed"/>
                        <Setter TargetName="Pulse" Property="Visibility" Value="Visible"/>
                        <Trigger.EnterActions>
                          <BeginStoryboard>
                            <Storyboard RepeatBehavior="Forever" AutoReverse="True">
                              <DoubleAnimation Storyboard.TargetName="Pulse" Storyboard.TargetProperty="Opacity" From="0.2" To="0.8" Duration="0:0:0.7"/>
                            </Storyboard>
                          </BeginStoryboard>
                        </Trigger.EnterActions>
                      </Trigger>
                    </ControlTemplate.Triggers>
                  </ControlTemplate>
                </ProgressBar.Template>
              </ProgressBar>
            </StackPanel>
          </Border>
          <ScrollViewer Grid.Row="2" VerticalScrollBarVisibility="Auto" Margin="0,12,0,0" x:Name="FeedScroll"><StackPanel x:Name="Feed"/></ScrollViewer>
          <Border Grid.Row="3" x:Name="ActionBox" Background="#3A3418" CornerRadius="10" Padding="16" Margin="0,12,0,0" Visibility="Collapsed">
            <StackPanel><TextBlock x:Name="ActionText" FontSize="15" Foreground="#FFD166" TextWrapping="Wrap"/>
              <StackPanel Orientation="Horizontal" Margin="0,12,0,0"><Button x:Name="BtnYes" Content="" Margin="0,0,10,0"/><Button x:Name="BtnNo" Content="" Background="#3A3F4A" Visibility="Collapsed"/></StackPanel></StackPanel>
          </Border>
        </Grid>
      </Border>
      <DockPanel Grid.Row="1" Grid.ColumnSpan="2" Margin="0,12,0,0">
        <Button x:Name="BtnStart" Content="" DockPanel.Dock="Left"/>
        <Button x:Name="BtnLog" Content="" Background="#2A2F3A" DockPanel.Dock="Left" Margin="10,0,0,0"/>
        <Button x:Name="BtnBack" Content="" Background="#2A2F3A" DockPanel.Dock="Left" Margin="10,0,0,0"/>
        <TextBlock x:Name="Status" Text="" Foreground="#8B939F" VerticalAlignment="Center" Margin="16,0,0,0"/>
      </DockPanel>
      <Border Grid.Row="0" Grid.ColumnSpan="2" x:Name="LogBox" Background="#0A0C10" CornerRadius="10" Padding="10" Visibility="Collapsed">
        <ScrollViewer VerticalScrollBarVisibility="Auto"><TextBox x:Name="Log" Background="Transparent" Foreground="#9AA3B0" BorderThickness="0" FontFamily="Consolas" FontSize="12" IsReadOnly="True" TextWrapping="NoWrap"/></ScrollViewer>
      </Border>
    </Grid>

    <!-- SEITE 3: Fertig -->
    <Border Grid.Row="1" x:Name="Page3" Background="#14181F" CornerRadius="14" Padding="30" Visibility="Collapsed">
      <StackPanel VerticalAlignment="Center" HorizontalAlignment="Center" MaxWidth="620">
        <TextBlock Text="✔" FontSize="56" Foreground="#34C759" HorizontalAlignment="Center"/>
        <TextBlock x:Name="DoneTitle" Text="" FontSize="26" FontWeight="SemiBold" Foreground="#F0F2F6" HorizontalAlignment="Center" Margin="0,8,0,0"/>
        <TextBlock x:Name="DoneSub" Text="" HorizontalAlignment="Center" Margin="0,6,0,18" TextWrapping="Wrap" TextAlignment="Center"/>
        <Border Background="#0A0C10" CornerRadius="10" Padding="16"><DockPanel><Button x:Name="BtnCopy" Content="" DockPanel.Dock="Right" Padding="12,6" Background="#2A2F3A" Margin="12,0,0,0"/><TextBox x:Name="DoneCmd" FontFamily="Consolas" FontSize="15" IsReadOnly="True" BorderThickness="0" Background="Transparent" Foreground="#F0F2F6" VerticalAlignment="Center"/></DockPanel></Border>
        <TextBlock x:Name="DoneHint" Text="" Margin="0,14,0,0" TextWrapping="Wrap" TextAlignment="Center" FontSize="13" Foreground="#8B939F"/>
        <Border x:Name="PkgPanel" Background="#0E1116" BorderBrush="#262A33" BorderThickness="1" CornerRadius="12" Padding="16" Margin="0,18,0,0" Visibility="Collapsed">
          <StackPanel>
            <TextBlock x:Name="PkgTitle" Style="{StaticResource Grp}"/>
            <TextBlock x:Name="PkgIntro" Style="{StaticResource Hint}"/>
            <CheckBox x:Name="PkgLlm" Margin="0,8,0,0" Foreground="#C9CFD8"/>
            <CheckBox x:Name="PkgStt" Margin="0,6,0,0" Foreground="#C9CFD8"/>
            <CheckBox x:Name="PkgPya" Margin="0,6,0,0" Foreground="#C9CFD8"/>
            <CheckBox x:Name="PkgTts" Margin="0,6,0,0" Foreground="#C9CFD8"/>
            <CheckBox x:Name="PkgWake" Margin="0,6,0,0" Foreground="#C9CFD8"/>
            <CheckBox x:Name="PkgImg" Margin="0,6,0,0" Foreground="#C9CFD8"/>
            <DockPanel Margin="0,12,0,0" LastChildFill="True">
              <Button x:Name="BtnPkg" Content="" DockPanel.Dock="Left" Padding="14,7"/>
              <TextBlock x:Name="PkgStatus" Text="" VerticalAlignment="Center" Margin="14,0,0,0" TextWrapping="Wrap" Foreground="#8B939F"/>
            </DockPanel>
          </StackPanel>
        </Border>
        <StackPanel Orientation="Horizontal" HorizontalAlignment="Center" Margin="0,22,0,0"><Button x:Name="BtnAgain" Content="" Margin="0,0,12,0"/><Button x:Name="BtnClose" Content="" Background="#2A2F3A"/></StackPanel>
      </StackPanel>
    </Border>
  </Grid>
</Window>
"@
$W = [Windows.Markup.XamlReader]::Load((New-Object System.Xml.XmlNodeReader $xaml))
$ui=@{}; foreach ($n in 'Logo','GrpAccess','GrpNet','GrpOp','SsidHint','HostHint','DispHint','Sub','Pager','FLang','Page1','Page2','Page3','P1Title','P1Intro','LUser','LPass','LPass2','LHost','LTz','LSsid','LPsk','LChg','LTo','LChgHint','LDisp','LKey','FUser','FPass','FPass2','FHost','FTz','FSsid','BtnWifi','FPsk','FKey','FKeyHint','FChgS','FChgE','FDisp','FErr','BtnNext1','BtnUnflash','Steps','StepTitle','StateBox','StateDot','StateText','Prog','ProgText','FeedScroll','Feed','ActionBox','ActionText','BtnYes','BtnNo','BtnStart','BtnLog','BtnBack','Status','LogBox','Log','DoneTitle','DoneSub','DoneCmd','BtnCopy','DoneHint','BtnAgain','BtnClose','PkgPanel','PkgTitle','PkgIntro','PkgLlm','PkgStt','PkgPya','PkgTts','PkgWake','PkgImg','BtnPkg','PkgStatus','PagePkg','PkTitle','PkIntro','CardLlm','ChkLlm','DescLlm','FactLlm','CardStt','ChkStt','DescStt','FactStt','CardPya','ChkPya','DescPya','FactPya','CardTts','ChkTts','DescTts','FactTts','CardWake','ChkWake','DescWake','FactWake','CardImg','ChkImg','DescImg','FactImg','PkCoexWarn','BtnPkgBack','BtnPkgNext','PageModel','MTitle','MIntro','MList','BtnMBack','BtnMNext','SvPkg') { $ui[$n]=$W.FindName($n) }

# Logo (barra-setup\logo.png) in Header + Fenster-Icon — fehlt die Datei, bleibt nur die Wortmarke
$logoFile = Join-Path $Kit 'logo.png'
if (Test-Path $logoFile) {
  try { $bi=New-Object Windows.Media.Imaging.BitmapImage; $bi.BeginInit(); $bi.UriSource=[Uri]$logoFile; $bi.EndInit(); $ui.Logo.Source=$bi; $W.Icon=$bi } catch {}
}

# ---------- Sprache anwenden (alle statischen Texte; wird auch beim Dropdown-Wechsel gerufen) ----------
$script:mode='setup'
function Apply-UiLang {
  # Versal-Label wie die dash2-Kacheltitel (WPF kennt kein text-transform -> ToUpper hier)
  $script:stateWords = @{ ready=(T 'setup.state.ready').ToUpper(); run=(T 'setup.state.run').ToUpper(); wait_user=(T 'setup.state.wait_user').ToUpper(); wait_dev=(T 'setup.state.wait_dev').ToUpper(); ok=(T 'setup.state.ok').ToUpper(); fail=(T 'setup.state.fail').ToUpper() }
  $ui.P1Title.Text=T 'setup.p1.title'; $ui.P1Intro.Text=T 'setup.p1.intro'
  $ui.LUser.Text=T 'setup.p1.user'; $ui.LPass.Text=T 'setup.p1.pass'; $ui.LPass2.Text=T 'setup.p1.pass2'
  $ui.LHost.Text=T 'setup.p1.host'; $ui.LTz.Text=T 'setup.p1.tz'; $ui.LSsid.Text=T 'setup.p1.ssid'
  $ui.BtnWifi.Content=T 'setup.p1.wifi_btn'; $ui.LPsk.Text=T 'setup.p1.psk'; $ui.LChg.Text=T 'setup.p1.charge'
  $ui.LTo.Text=(T 'setup.p1.to')+'  '; $ui.LChgHint.Text=T 'setup.p1.charge_hint'; $ui.LDisp.Text=T 'setup.p1.disp'; $ui.LKey.Text=T 'setup.p1.key'
  $ui.GrpAccess.Text=T 'setup.p1.grp_access'; $ui.GrpNet.Text=T 'setup.p1.grp_net'; $ui.GrpOp.Text=T 'setup.p1.grp_op'
  $ui.SsidHint.Text=T 'setup.p1.ssid_hint'; $ui.HostHint.Text=T 'setup.p1.host_hint'; $ui.DispHint.Text=T 'setup.p1.disp_hint'
  $ui.BtnUnflash.Content=T 'setup.p1.unflash'; $ui.BtnUnflash.ToolTip=T 'setup.p1.unflash_tip'; $ui.BtnNext1.Content=T 'setup.p1.next'
  $ui.StepTitle.Text=T 'setup.p2.title0'; $ui.StateText.Text=$script:stateWords.ready
  $ui.BtnYes.Content=T 'setup.p2.done_btn'; $ui.BtnNo.Content=T 'setup.p2.cancel'
  $ui.BtnStart.Content=T 'setup.p2.start'; $ui.BtnLog.Content=T 'setup.p2.details_closed'; $ui.BtnBack.Content=T 'setup.p2.back'
  $ui.DoneTitle.Text=T 'setup.p3.done'; $ui.BtnCopy.Content=T 'setup.p3.copy'; $ui.BtnAgain.Content=T 'setup.p3.again'; $ui.BtnClose.Content=T 'setup.p3.close'
  $ui.PkgTitle.Text=(T 'setup.p3.pkg_title').ToUpper(); $ui.PkgIntro.Text=T 'setup.p3.pkg_intro'
  $ui.PkgLlm.Content=T 'setup.p3.pkg_llm'; $ui.PkgStt.Content=T 'setup.p3.pkg_stt'; $ui.PkgPya.Content=T 'setup.p3.pkg_pya'; $ui.PkgTts.Content=T 'setup.p3.pkg_tts'; $ui.PkgWake.Content=T 'setup.p3.pkg_wake'; $ui.PkgImg.Content=T 'setup.p3.pkg_img'; $ui.BtnPkg.Content=T 'setup.p3.pkg_btn'
  # Pakete-Bildschirm (Seite 1b) + Modellwahl (Seite 1c)
  $ui.PkTitle.Text=T 'setup.pkg.title'; $ui.PkIntro.Text=T 'setup.pkg.intro'
  $ui.ChkLlm.Content=T 'setup.pkg.llm_name'; $ui.DescLlm.Text=T 'setup.pkg.llm_desc'; $ui.FactLlm.Text=T 'setup.pkg.llm_fact'
  $ui.ChkStt.Content=T 'setup.pkg.stt_name'; $ui.DescStt.Text=T 'setup.pkg.stt_desc'; $ui.FactStt.Text=T 'setup.pkg.stt_fact'
  $ui.ChkPya.Content=T 'setup.pkg.pya_name'; $ui.DescPya.Text=T 'setup.pkg.pya_desc'; $ui.FactPya.Text=T 'setup.pkg.pya_fact'
  $ui.ChkTts.Content=T 'setup.pkg.tts_name'; $ui.DescTts.Text=T 'setup.pkg.tts_desc'; $ui.FactTts.Text=T 'setup.pkg.tts_fact'
  $ui.ChkWake.Content=T 'setup.pkg.wake_name'; $ui.DescWake.Text=T 'setup.pkg.wake_desc'; $ui.FactWake.Text=T 'setup.pkg.wake_fact'
  $ui.ChkImg.Content=T 'setup.pkg.img_name'; $ui.DescImg.Text=T 'setup.pkg.img_desc'; $ui.FactImg.Text=T 'setup.pkg.img_fact'
  $ui.PkCoexWarn.Text=T 'setup.pkg.coex_warn'
  $ui.BtnPkgBack.Content=T 'setup.p2.back'; $ui.BtnPkgNext.Content=T 'setup.p1.next'
  $ui.MIntro.Text=T 'setup.model.intro'; $ui.BtnMBack.Content=T 'setup.p2.back'; $ui.BtnMNext.Content=T 'setup.p1.next'
  Set-Mode $script:mode
}

# ---------- Seite 1: Vorbelegung + Windows-WLAN ----------
function Get-CurrentWifi {
  try {
    $o = netsh wlan show interfaces 2>$null | Out-String
    if ($o -match '(?m)^\s*(SSID|Name)\s*:\s*(.+)$') { $ssid = ($o | Select-String '(?m)^\s*SSID\s*:\s*(.+)$' | Select-Object -First 1).Matches[0].Groups[1].Value.Trim() }
    if (-not $ssid) { return $null }
    $p = netsh wlan show profile name="$ssid" key=clear 2>$null | Out-String
    $psk = ($p | Select-String '(?m)^\s*(Schlüsselinhalt|Key Content)\s*:\s*(.+)$' | Select-Object -First 1)
    $key = if ($psk) { $psk.Matches[0].Groups[2].Value.Trim() } else { '' }
    return @{ ssid=$ssid; psk=$key }
  } catch { return $null }
}
$ui.FUser.Text = if ($saved -and $saved.user) { $saved.user } else { ($env:USERNAME -replace '[^a-z0-9_-]','').ToLower() }
# Hostname: automatischer 'barra-xxxx' wird JEDES Mal neu gewuerfelt (jede Node eindeutig); nur ein selbst
# vergebener Name (nicht dem Muster entsprechend) wird aus der gespeicherten Konfiguration uebernommen.
function New-AutoHost(){ 'barra-' + (-join (1..4 | ForEach-Object { [char](Get-Random -InputObject (48..57+97..102)) })) }
$ui.FHost.Text = if ($saved -and $saved.host -and $saved.host -notmatch '^barra-[0-9a-f]{4}$') { $saved.host } else { New-AutoHost }
$ui.FTz.Text   = if ($saved -and $saved.tz) { $saved.tz } else { try { $tz=(Get-TimeZone).Id; $m=@{ 'W. Europe Standard Time'='Europe/Berlin'; 'Central European Standard Time'='Europe/Warsaw'; 'GMT Standard Time'='Europe/London'; 'UTC'='UTC' }; if ($m[$tz]) { $m[$tz] } else { 'Europe/Berlin' } } catch { 'Europe/Berlin' } }
$ui.FSsid.Text = if ($saved -and $saved.ssid) { $saved.ssid } else { '' }
if ($saved -and $saved.psk) { $ui.FPsk.Password = $saved.psk }
$pub = Get-ChildItem "$env:USERPROFILE\.ssh\id_*.pub" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($saved -and $saved.key) { $ui.FKey.Text = $saved.key } elseif ($pub) { $ui.FKey.Text = (Get-Content $pub.FullName -Raw).Trim(); $ui.FKeyHint.Text = (T 'setup.p1.key_found' $pub.Name) }
if ($saved -and $saved.chgS) { $ui.FChgS.Text=$saved.chgS; $ui.FChgE.Text=$saved.chgE; $ui.FDisp.Text=$saved.disp }
$ui.BtnWifi.Add_Click({ $w = Get-CurrentWifi; if ($w) { $ui.FSsid.Text=$w.ssid; if ($w.psk) { $ui.FPsk.Password=$w.psk }; $ui.FErr.Text='' } else { $ui.FErr.Text=T 'setup.err.nowifi' } })
if (-not $ui.FSsid.Text) { $w0 = Get-CurrentWifi; if ($w0) { $ui.FSsid.Text=$w0.ssid; if ($w0.psk) { $ui.FPsk.Password=$w0.psk } } }

# Sprach-Dropdown
$langNames = @{ de='Deutsch'; en='English' }
foreach ($code in (Get-BarraI18nLanguages (Join-Path $Kit 'i18n'))) {
  $item = New-Object Windows.Controls.ComboBoxItem
  $item.Content = if ($langNames[$code]) { $langNames[$code] } else { $code }
  $item.Tag = $code
  [void]$ui.FLang.Items.Add($item)
  if ($code -eq $script:lang) { $ui.FLang.SelectedItem = $item }
}
$ui.FLang.Add_SelectionChanged({
  $sel = $ui.FLang.SelectedItem; if (-not $sel) { return }
  if ($sel.Tag -ne $script:lang) {
    $script:lang = $sel.Tag
    Initialize-BarraI18n -Dir (Join-Path $Kit 'i18n') -Lang $script:lang
    Apply-UiLang
  }
})

# ---------- Seiten ----------
# 1=Formular, 'pkg'=Paket-Auswahl, 'model'=Modellwahl, 2=Flashen, 3=Fertig
function Show-Page($n){
  $ui.Page1.Visibility   = if($n -eq 1){'Visible'}else{'Collapsed'}
  $ui.PagePkg.Visibility = if($n -eq 'pkg'){'Visible'}else{'Collapsed'}
  $ui.PageModel.Visibility = if($n -eq 'model'){'Visible'}else{'Collapsed'}
  $ui.Page2.Visibility   = if($n -eq 2){'Visible'}else{'Collapsed'}
  $ui.Page3.Visibility   = if($n -eq 3){'Visible'}else{'Collapsed'}
  $ui.Pager.Text = switch ($n) { 'pkg' { T 'setup.pager_pkg' } 'model' { T 'setup.pager_model' } default { T 'setup.pager' $n } }
}

# ---------- Seite 2: Schritte + Zustand ----------
$script:stepNames=@(); $script:stepUI=@()
function Set-StepNames($names){
  $script:stepNames=$names; $script:stepUI=@(); $ui.Steps.Children.Clear()
  foreach ($s in $names) { $tb=New-Object Windows.Controls.TextBlock; $tb.Text="○  $s"; $tb.FontSize=14; $tb.Foreground='#8B939F'; $tb.Margin='0,7,0,7'; $ui.Steps.Children.Add($tb)|Out-Null; $script:stepUI+=$tb }
}
function Set-Mode($m){
  $script:mode=$m
  $setupSteps  =@((T 'setup.steps.connect'),(T 'setup.steps.unlock'),(T 'setup.steps.stock'),(T 'setup.steps.kernel'),(T 'setup.steps.payload'),(T 'setup.steps.firstboot'))
  $unflashSteps=@((T 'setup.steps.connect'),(T 'setup.steps.u_bl'),(T 'setup.steps.u_stock'),(T 'setup.steps.u_lock'))
  if ($m -eq 'unflash') { Set-StepNames $unflashSteps; $ui.Sub.Text=T 'setup.sub_unflash' } else { Set-StepNames $setupSteps; $ui.Sub.Text=T 'setup.sub' }
}

$stateColors=@{ run='#5B8DEF'; wait_user='#FFD166'; wait_dev='#FF9F0A'; ok='#34C759'; fail='#FF453A' }
$script:curStep=-1; $script:lastState=''
function Set-Step($n,$title,$state){
  $stepUI=$script:stepUI; $stepNames=$script:stepNames
  if ($n -ge $stepUI.Count) { $n=$stepUI.Count-1 }
  for ($k=0;$k -lt $stepUI.Count;$k++){ if ($k -lt $n){$stepUI[$k].Text="●  "+$stepNames[$k];$stepUI[$k].Foreground='#34C759';$stepUI[$k].FontWeight='Normal'} elseif ($k -eq $n){ $stepUI[$k].Text="▶  "+$stepNames[$k]; $stepUI[$k].Foreground=$(if($state -eq 'fail'){'#FF453A'}else{'#F0F2F6'}); $stepUI[$k].FontWeight='SemiBold' } else {$stepUI[$k].Text="○  "+$stepNames[$k];$stepUI[$k].Foreground='#8B939F';$stepUI[$k].FontWeight='Normal'} }
  if ($state -eq 'ok') { $stepUI[$n].Text="●  "+$stepNames[$n]; $stepUI[$n].Foreground='#34C759' }
  $ui.StepTitle.Text=$title; $ui.StateDot.Foreground=$stateColors[$state]; $ui.StateText.Text=$script:stateWords[$state]
  if ($state -ne 'wait_user') { $ui.ActionBox.Visibility='Collapsed' }
  if ($state -in 'ok','fail') { $ui.Prog.Visibility='Collapsed'; $ui.ProgText.Text='' }
  $script:curStep=$n; $script:lastState=$state
}
function Add-Feed($text,$color,$prefix=''){ Write-LogFile ("## " + (Get-Date -Format 'HH:mm:ss') + " $prefix$text"); $tb=New-Object Windows.Controls.TextBlock; $tb.Text="$prefix$text"; $tb.FontSize=14; $tb.Foreground=$color; $tb.TextWrapping='Wrap'; $tb.Margin='0,3,0,3'; $ui.Feed.Children.Add($tb)|Out-Null; if($ui.Feed.Children.Count -gt 80){$ui.Feed.Children.RemoveAt(0)}; $ui.FeedScroll.ScrollToEnd() }
# Protokoll immer auf Platte (fuer Diagnose): %LOCALAPPDATA%\barra\setup.log (Vorlauf -> setup.prev.log)
$script:LogFile = Join-Path $env:LOCALAPPDATA 'barra\setup.log'
try { New-Item -ItemType Directory -Force (Split-Path $script:LogFile) | Out-Null
      if (Test-Path $script:LogFile) { Move-Item $script:LogFile ($script:LogFile -replace 'setup\.log$','setup.prev.log') -Force }
      [IO.File]::WriteAllText($script:LogFile, "barra Setup $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')`r`n", [Text.Encoding]::UTF8) } catch {}
function Write-LogFile($t){ try { [IO.File]::AppendAllText($script:LogFile, "$t`r`n", [Text.Encoding]::UTF8) } catch {} }
function Add-Log($t){ Write-LogFile $t; $ui.Log.AppendText("$t`r`n"); if ($ui.Log.Text.Length -gt 400000) { $ui.Log.Text=$ui.Log.Text.Substring(200000) }; $ui.Log.ScrollToEnd() }

Apply-UiLang
Show-Page 1

# ---------- Kern im Runspace ----------
$sync=[hashtable]::Synchronized(@{ q=[System.Collections.Concurrent.ConcurrentQueue[object]]::new(); answer=$null; cancel=$false; done=$false; running=$false })
$script:rs=$null; $script:ps=$null
function Start-Core($precfg,$coreMode=$null){
  $sync.done=$false; $sync.running=$true; $sync.cancel=$false
  $script:rs=[runspacefactory]::CreateRunspace(); $script:rs.ApartmentState='MTA'; $script:rs.Open()
  # ACHTUNG: PowerShell-Variablen sind case-insensitiv und der Kern setzt beim Laden $script:PreCfg/$script:Kit
  # in DERSELBEN Scope -> die injizierten Variablen brauchen eindeutige Namen (sonst genullt).
  $modeIn = if ($coreMode) { $coreMode } else { $script:mode }
  $script:rs.SessionStateProxy.SetVariable('sync',$sync); $script:rs.SessionStateProxy.SetVariable('barraKitDir',$Kit); $script:rs.SessionStateProxy.SetVariable('barraPreCfgIn',$precfg); $script:rs.SessionStateProxy.SetVariable('barraModeIn',$modeIn); $script:rs.SessionStateProxy.SetVariable('barraLangIn',$script:lang)
  $script:ps=[powershell]::Create(); $script:ps.Runspace=$script:rs
  $script:ps.AddScript({
    . (Join-Path $barraKitDir 'barra-i18n.ps1')
    Initialize-BarraI18n -Dir (Join-Path $barraKitDir 'i18n') -Lang $barraLangIn
    . (Join-Path $barraKitDir 'barra-core.ps1')
    $script:Emit = { param($k,$d) $sync.q.Enqueue(@{ k=$k; d=$d }) }
    $script:PreCfg = $barraPreCfgIn
    # Answer/Cancel aus der GUI durchreichen
    $script:Answer = $sync.answerQ
    # F7: kooperativen Abbruch wirklich verdrahten — der Kern pollt diesen Hook (Chk/Ask/Run)
    # und setzt $script:Cancel, sodass laufende adb/fastboot-Kinder gekillt werden.
    $script:CancelCheck = { if ($sync.cancel) { $script:Cancel = $true } }
    $t = [System.Threading.Thread]::CurrentThread
    try { if ($barraModeIn -eq 'unflash') { Run-Unflash } elseif ($barraModeIn -eq 'kits') { Run-Kits } else { Run-Flash } } catch { $sync.q.Enqueue(@{ k='fail'; d=(T 'core.corefail' "$_") }) }
    $sync.done=$true
  }) | Out-Null
  $sync.answerQ=[System.Collections.Concurrent.ConcurrentQueue[string]]::new()
  $script:ps.BeginInvoke() | Out-Null
}
$script:askKind=''
function Send-Answer($a){ Write-LogFile "## Antwort: $a"; $sync.answerQ.Enqueue($a); $ui.ActionBox.Visibility='Collapsed'
  # Zustandsanzeige zuruecksetzen — sonst bleibt "Warte auf Nutzer" stehen, bis der naechste
  # step-Event kommt (der kann bei langen Phasen wie Download/Flash minutenlang ausbleiben)
  $st = if ($script:lastState -and $script:lastState -ne 'wait_user') { $script:lastState } else { 'run' }
  $ui.StateDot.Foreground=$stateColors[$st]; $ui.StateText.Text=$script:stateWords[$st] }

$ui.BtnYes.Add_Click({ Send-Answer $(if($script:askKind -eq 'yn'){'j'}else{'ok'}) })
$ui.BtnNo.Add_Click({ Send-Answer 'n' })
$ui.BtnLog.Add_Click({ if($ui.LogBox.Visibility -eq 'Visible'){$ui.LogBox.Visibility='Collapsed';$ui.BtnLog.Content=T 'setup.p2.details_closed'}else{$ui.LogBox.Visibility='Visible';$ui.BtnLog.Content=T 'setup.p2.details_open'} })
$ui.BtnBack.Add_Click({ if (-not $sync.running) { Set-Mode 'setup'; $ui.BtnStart.Content=T 'setup.p2.start'; $ui.BtnStart.IsEnabled=$true; Show-Page 1 } })
$ui.BtnUnflash.Add_Click({
  Set-Mode 'unflash'; $ui.FErr.Text=''; $ui.Feed.Children.Clear(); $ui.BtnStart.Content=T 'setup.p2.start'; $ui.BtnStart.IsEnabled=$true; $ui.Status.Text=''
  Set-Step 0 (T 'setup.p2.unflash_title') 'wait_dev'
  Show-Page 2; $ui.Pager.Text=T 'setup.pager_unflash'
  Add-Feed (T 'setup.p2.unflash_warn') '#FFB020' '⚠  '
  Add-Feed (T 'setup.p2.unflash_hint') '#C9CFD8'
})
$ui.BtnStart.Add_Click({
  $ui.BtnStart.IsEnabled=$false; $ui.BtnStart.Content=T 'setup.p2.running'; $ui.BtnBack.IsEnabled=$false; $ui.Status.Text=''
  $ui.Feed.Children.Clear(); Set-Step 0 (T 'setup.steps.connect') 'wait_dev'
  Start-Core $script:precfg
})

# ---------- Seite 1 → 2 ----------
$ui.BtnNext1.Add_Click({
  $u=$ui.FUser.Text.Trim(); $p1=$ui.FPass.Password; $p2=$ui.FPass2.Password; $h=$ui.FHost.Text.Trim(); $tz=$ui.FTz.Text.Trim(); $ssid=$ui.FSsid.Text.Trim(); $psk=$ui.FPsk.Password; $key=$ui.FKey.Text.Trim()
  if ($u -notmatch '^[a-z][a-z0-9_-]{0,31}$') { $ui.FErr.Text=T 'setup.err.user'; return }
  if ($p1.Length -lt 4) { $ui.FErr.Text=T 'setup.err.pass'; return }
  if ($p1 -ne $p2) { $ui.FErr.Text=T 'setup.err.pass2'; return }
  if ($h -notmatch '^[a-z0-9][a-z0-9-]{0,62}$') { $ui.FErr.Text=T 'setup.err.host'; return }
  if ($ssid -and $psk -and $psk.Length -lt 8) { $ui.FErr.Text=T 'setup.err.psk'; return }
  $ui.FErr.Text=''
  # Passwort geht in die preconfig (wird auf dem Geraet als root per chpasswd gesetzt; Datei danach geloescht)
  # SSH-Key(s) base64-kodiert transportieren: erhaelt mehrere Zeilen (mehrere Keys) und ist
  # injektionssicher (kein Newline->Space-Flattening, keine Quote-Ausbrueche auf dem Geraet).
  $keyB64 = if ($key) { [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes(($key -replace "`r`n","`n"))) } else { '' }
  $script:precfg = [ordered]@{ USER=$u; PASS=$p1; HOST=$h; TZ=$tz; SSID=$ssid; PSK=$psk; SSHKEY_B64=$keyB64; CHARGE_START=$ui.FChgS.Text; CHARGE_STOP=$ui.FChgE.Text; DISPLAY_TIMEOUT=$ui.FDisp.Text; LANG_UI=$script:lang }
  Save-Cfg @{ user=$u; host=$h; tz=$tz; ssid=$ssid; psk=$psk; key=$key; chgS=$ui.FChgS.Text; chgE=$ui.FChgE.Text; disp=$ui.FDisp.Text; lang=$script:lang }
  # Weiter zum Paket-Bildschirm (falls Kits vorliegen), sonst direkt zum Flashen
  Detect-Kits
  if ($script:kitAvail.Count) { Init-Pkg-Screen; Show-Page 'pkg' } else { Enter-FlashPage @() }
})

# ---------- Pakete-Bildschirm (1b) + Modellwahl (1c) ----------
# Datengetriebener Katalog: nur Kits/Modelle anbieten, deren Dateien im Setup-Ordner liegen.
$script:KitCatalog = @(
  @{ id='llm'; name={T 'setup.pkg.llm_name'}; models=@(
      @{ id='qwen3-4b';     name={T 'setup.model.llm_qwen4b'};    desc={T 'setup.model.llm_qwen4b_desc'};    files=@('llm-kit\llm-attn-qwen3-4b.tar','llm-kit\qwen3-4b.gguf') }
      @{ id='qwen2.5-1.5b'; name={T 'setup.model.llm_qwen15b'};   desc={T 'setup.model.llm_qwen15b_desc'};   files=@('llm-kit\llm-attn-qwen2.5-1.5b.tar','llm-kit\qwen2.5-1.5b.gguf') }
      @{ id='glm-edge-4b';  name={T 'setup.model.llm_glmedge'};   desc={T 'setup.model.llm_glmedge_desc'};   files=@('llm-kit\llm-attn-glm-edge-4b-chat.tar','llm-kit\glm-edge-4b-chat.gguf') }
      @{ id='qwen38-distill'; name={T 'setup.model.llm_qwen38d'}; desc={T 'setup.model.llm_qwen38d_desc'};   files=@('llm-kit\qwen38-4b-distill.gguf') }
      @{ id='gemma-e2b';    name={T 'setup.model.llm_gemma_e2b'}; desc={T 'setup.model.llm_gemma_e2b_desc'}; files=@('llm-kit\llm-attn-gemma-4-e2b-q4_0.tar','llm-kit\gemma-4-e2b-q4_0.gguf') }
      @{ id='gemma-e4b';    name={T 'setup.model.llm_gemma_e4b'}; desc={T 'setup.model.llm_gemma_e4b_desc'}; files=@('llm-kit\gemma-4-e4b-q3_k_s.gguf') } ) }
  @{ id='stt'; name={T 'setup.pkg.stt_name'}; models=@(
      @{ id='turbo';  name={T 'setup.model.stt_turbo'};  desc={T 'setup.model.stt_turbo_desc'};  files=@('whisper-kit\whisper-kit-turbo.tar','whisper-kit\ggml-large-v3-turbo-q5_0.bin') }
      @{ id='medium'; name={T 'setup.model.stt_medium'}; desc={T 'setup.model.stt_medium_desc'}; files=@('whisper-kit\whisper-kit-medium.tar','whisper-kit\ggml-medium-q5_0.bin') }
      @{ id='small';  name={T 'setup.model.stt_small'};  desc={T 'setup.model.stt_small_desc'};  files=@('whisper-kit\whisper-kit-small.tar','whisper-kit\ggml-small.bin') }
      @{ id='base';   name={T 'setup.model.stt_base'};   desc={T 'setup.model.stt_base_desc'};   files=@('whisper-kit\whisper-kit-base.tar','whisper-kit\ggml-base.bin') }
      @{ id='tiny';   name={T 'setup.model.stt_tiny'};   desc={T 'setup.model.stt_tiny_desc'};   files=@('whisper-kit\whisper-kit-tiny.tar','whisper-kit\ggml-tiny.bin') } ) }
  @{ id='pya'; name={T 'setup.pkg.pya_name'}; models=@(
      @{ id='resnet34-en'; name={T 'setup.model.pya_r34'};     desc={T 'setup.model.pya_r34_desc'};     files=@('pyannote-kit\pyannote-kit.tar') }
      @{ id='resnet34-zh'; name={T 'setup.model.pya_r34zh'};   desc={T 'setup.model.pya_r34zh_desc'};   files=@('pyannote-kit\pyannote-kit.tar','pyannote-kit\r34zh_trunk.package','pyannote-kit\head_zh.bin','pyannote-kit\resnet34_zh.onnx') }
      @{ id='eres2net-zh'; name={T 'setup.model.pya_eres'};    desc={T 'setup.model.pya_eres_desc'};    files=@('pyannote-kit\pyannote-kit.tar','pyannote-kit\eres_body.package','pyannote-kit\eres_tail.onnx','pyannote-kit\head_eres.bin','pyannote-kit\eres_params.txt') }
      @{ id='titanet-en';  name={T 'setup.model.pya_titanet'}; desc={T 'setup.model.pya_titanet_desc'}; files=@('pyannote-kit\pyannote-kit.tar','pyannote-kit\tita_seg0.package','pyannote-kit\tita_seg4.package','pyannote-kit\tita_tail.onnx','pyannote-kit\tita_glue.bin','pyannote-kit\tita_params.txt') } ) }
  @{ id='tts'; name={T 'setup.pkg.tts_name'}; models=@(
      @{ id='all'; name={T 'setup.model.tts_all'}; desc={T 'setup.model.tts_all_desc'}; files=@('tts-kit\tts-kit.tar.gz') } ) }
  @{ id='wake'; name={T 'setup.pkg.wake_name'}; models=@(
      @{ id='heybarra'; name={T 'setup.model.wake_heybarra'}; desc={T 'setup.model.wake_heybarra_desc'}; files=@('wake-kit\wake-kit.tar.gz') } ) }
  @{ id='img'; name={T 'setup.pkg.img_name'}; models=@(
      @{ id='dreamshaper-lcm'; name={T 'setup.model.img_lcm'}; desc={T 'setup.model.img_lcm_desc'}; files=@('img-kit\img-kit.tar.gz','img-kit\DreamShaper8_LCM.safetensors','img-kit\taesd.safetensors') } ) }
)
function Get-KitModels($kitId){
  $k = $script:KitCatalog | Where-Object { $_.id -eq $kitId }
  if (-not $k) { return ,@() }
  # ,@(...): PowerShell entrollt Ein-Element-Arrays beim Return — das Komma erhaelt das Array
  ,@($k.models | Where-Object { $m=$_; -not @($m.files | Where-Object { -not (Test-Path (Join-Path $Kit $_)) }).Count })
}
function Update-CoexWarn {
  $llm = $ui.ChkLlm.IsEnabled -and $ui.ChkLlm.IsChecked
  $other = ($ui.ChkStt.IsEnabled -and $ui.ChkStt.IsChecked) -or ($ui.ChkPya.IsEnabled -and $ui.ChkPya.IsChecked)
  $ui.PkCoexWarn.Visibility = if ($llm -and $other) { 'Visible' } else { 'Collapsed' }
}
function Init-Pkg-Screen {
  foreach ($row in @(@('llm',$ui.CardLlm,$ui.ChkLlm),@('stt',$ui.CardStt,$ui.ChkStt),@('pya',$ui.CardPya,$ui.ChkPya),@('tts',$ui.CardTts,$ui.ChkTts),@('wake',$ui.CardWake,$ui.ChkWake),@('img',$ui.CardImg,$ui.ChkImg))) {
    $have = [bool](Get-KitModels $row[0]).Count
    $row[1].Visibility = if ($have) { 'Visible' } else { 'Collapsed' }
    $row[2].IsEnabled = $have; if (-not $have) { $row[2].IsChecked = $false }
  }
  Update-CoexWarn
  $ui.SvPkg.ScrollToTop()
}
foreach ($cb in @($ui.ChkLlm,$ui.ChkStt,$ui.ChkPya,$ui.ChkTts,$ui.ChkWake,$ui.ChkImg)) { $cb.Add_Checked({ Update-CoexWarn }); $cb.Add_Unchecked({ Update-CoexWarn }) }
$ui.BtnPkgBack.Add_Click({ Show-Page 1 })
$ui.BtnPkgNext.Add_Click({
  $sel=@()
  if ($ui.ChkLlm.IsEnabled -and $ui.ChkLlm.IsChecked) { $sel+='llm' }
  if ($ui.ChkStt.IsEnabled -and $ui.ChkStt.IsChecked) { $sel+='stt' }
  if ($ui.ChkPya.IsEnabled -and $ui.ChkPya.IsChecked) { $sel+='pya' }
  if ($ui.ChkTts.IsEnabled -and $ui.ChkTts.IsChecked) { $sel+='tts' }
  if ($ui.ChkWake.IsEnabled -and $ui.ChkWake.IsChecked) { $sel+='wake' }
  if ($ui.ChkImg.IsEnabled -and $ui.ChkImg.IsChecked) { $sel+='img' }
  $script:selKits=$sel; $script:kitModels=@{}
  if ($sel.Count) { $script:modelIdx=0; Show-Model-Screen } else { Enter-FlashPage @() }
})
function Show-Model-Screen {
  $kitId = $script:selKits[$script:modelIdx]
  $k = $script:KitCatalog | Where-Object { $_.id -eq $kitId }
  $ui.MTitle.Text = T 'setup.model.title' (& $k.name)
  $ui.MList.Children.Clear(); $script:modelRadios=@()
  $pre = $script:kitModels[$kitId]
  $models = Get-KitModels $kitId
  for ($i=0; $i -lt $models.Count; $i++) {
    $m = $models[$i]
    $rb = New-Object Windows.Controls.RadioButton
    $rb.GroupName='mdl'; $rb.Tag=$m.id; $rb.FontSize=15; $rb.FontWeight='SemiBold'; $rb.Foreground='#F0F2F6'
    $rb.Content = (& $m.name)
    $rb.IsChecked = if ($pre) { $m.id -eq $pre } else { $i -eq 0 }
    $desc = New-Object Windows.Controls.TextBlock
    $desc.Text = (& $m.desc); $desc.TextWrapping='Wrap'; $desc.FontSize=13; $desc.Foreground='#8B939F'; $desc.Margin='26,4,0,0'
    $card = New-Object Windows.Controls.Border
    $card.Background='#0E1116'; $card.BorderBrush='#262A33'; $card.BorderThickness=1; $card.CornerRadius=12; $card.Padding='18,14,18,14'; $card.Margin='0,0,0,12'
    $sp = New-Object Windows.Controls.StackPanel; [void]$sp.Children.Add($rb); [void]$sp.Children.Add($desc); $card.Child=$sp
    [void]$ui.MList.Children.Add($card); $script:modelRadios+=$rb
  }
  Show-Page 'model'
}
$ui.BtnMBack.Add_Click({ if ($script:modelIdx -gt 0) { $script:modelIdx--; Show-Model-Screen } else { Show-Page 'pkg' } })
$ui.BtnMNext.Add_Click({
  $kitId = $script:selKits[$script:modelIdx]
  $chosen = $script:modelRadios | Where-Object { $_.IsChecked } | Select-Object -First 1
  $script:kitModels[$kitId] = if ($chosen) { $chosen.Tag } else { (Get-KitModels $kitId)[0].id }
  if ($script:modelIdx -lt $script:selKits.Count-1) { $script:modelIdx++; Show-Model-Screen }
  else { Enter-FlashPage $script:selKits }
})
function Enter-FlashPage($selKits){
  if ($selKits.Count) { $script:precfg.KITS=$selKits; $script:precfg.KITMODELS=$script:kitModels }
  else { $script:precfg.Remove('KITS'); $script:precfg.Remove('KITMODELS') }
  Show-Page 2
  if ($selKits.Count) {
    Set-StepNames @((T 'setup.steps.connect'),(T 'setup.steps.unlock'),(T 'setup.steps.stock'),(T 'setup.steps.kernel'),(T 'setup.steps.payload'),(T 'setup.steps.firstboot'),(T 'setup.steps.packages'))
  }
  Add-Feed (T 'setup.p2.saved' $script:precfg.USER $script:precfg.HOST $(if($script:precfg.SSID){ T 'setup.p2.saved_wifi' $script:precfg.SSID } else { '' })) '#8B939F'
  Add-Feed (T 'setup.p2.connect_hint') '#C9CFD8'
}

# ---------- Ereignisse ----------
$timer=New-Object Windows.Threading.DispatcherTimer; $timer.Interval=[TimeSpan]::FromMilliseconds(120)
$timer.Add_Tick({
  $e=$null
  while ($sync.q.TryDequeue([ref]$e)) {
    $k=$e.k; $d=$e.d
    try {
    switch ($k) {
      'log'   { Add-Log $d }
      'step'  { Set-Step $d.n $d.title $d.state }
      'info'  { Add-Feed $d '#C9CFD8' }
      'ok'    { Add-Feed $d '#34C759' '✔  ' }
      'warn'  { Add-Feed $d '#FFB020' '⚠  ' }
      'fail'  { Add-Feed $d '#FF453A' '✘  '; $ui.StateDot.Foreground='#FF453A'; $ui.StateText.Text=$script:stateWords.fail; $script:lastState='fail'; $ui.Prog.Visibility='Collapsed'; $ui.ProgText.Text=''; if ($script:curStep -ge 0 -and $script:curStep -lt $script:stepUI.Count) { $script:stepUI[$script:curStep].Foreground='#FF453A' } }
      'tel'   { Add-Feed $d '#FFD166' '📱  ' }
      'ask'   { $script:askKind=$d.id; $ui.ActionText.Text=$d.text; if ($d.id -eq 'yn') { $ui.BtnYes.Content=T 'setup.p2.yes'; $ui.BtnNo.Visibility='Visible' } else { $ui.BtnYes.Content=T 'setup.p2.done_btn'; $ui.BtnNo.Visibility='Collapsed' }; $ui.ActionBox.Visibility='Visible'; $ui.StateDot.Foreground='#FFD166'; $ui.StateText.Text=$script:stateWords.wait_user }
      'progress' { if ($d.pct -ge 0) { $ui.Prog.Visibility='Visible'; $ui.Prog.IsIndeterminate=$false; $ui.Prog.Value=$d.pct } else { $ui.Prog.Visibility='Visible'; $ui.Prog.IsIndeterminate=$true }; $ui.ProgText.Text=$d.text }
      'done'  { if ($d.kind -eq 'unflash') { $ui.DoneTitle.Text=T 'setup.p3.u_title'; $ui.DoneSub.Text=T 'setup.p3.u_sub'; $ui.DoneCmd.Text=T 'setup.p3.u_cmd'; $ui.DoneHint.Text=T 'setup.p3.u_hint'; $ui.PkgPanel.Visibility='Collapsed'; Show-Page 3; break }
                $ui.DoneTitle.Text=(T 'setup.p3.done_host' $d.host); $ui.DoneSub.Text=$(if($d.ip){ T 'setup.p3.reach' $d.ip }else{ T 'setup.p3.noip' }); $ui.DoneCmd.Text=$(if($d.ip){"ssh $($d.user)@$($d.ip)"}else{"ssh $($d.user)@<ip>"}); $ui.DoneHint.Text=T 'setup.p3.hint'; Show-Pkg-Panel $true; Show-Page 3 }
      'pkg'   { $ui.PkgStatus.Text=$d.text
                $ui.PkgStatus.Foreground=$(if($d.state -eq 'ok'){'#34C759'}elseif($d.state -eq 'fail'){'#FF453A'}else{'#8B939F'})
                if ($d.state -in 'ok','fail') { $ui.BtnPkg.IsEnabled=$true; $ui.PkgLlm.IsEnabled=$script:kitLlm; $ui.PkgStt.IsEnabled=$script:kitStt; $ui.PkgPya.IsEnabled=$script:kitPya; $ui.PkgTts.IsEnabled=$script:kitTts; $ui.PkgWake.IsEnabled=$script:kitWake; $ui.PkgImg.IsEnabled=$script:kitImg } }
    }
    } catch { Write-LogFile "GUI-Fehler bei '$k': $_" }
  }
  if ($sync.done -and $sync.running) { $sync.running=$false; $ui.BtnStart.IsEnabled=$true; $ui.BtnStart.Content=T 'setup.p2.restart'; $ui.BtnBack.IsEnabled=$true; if ($script:lastState -ne 'ok') { $ui.Status.Text=T 'setup.p2.ended' } }
})
$timer.Start()
# ---------- Pakete: Verfuegbarkeit + Fertig-Seiten-Panel (nachtraegliche Installation) ----------
$script:kitLlm=$false; $script:kitStt=$false; $script:kitPya=$false; $script:kitTts=$false; $script:kitWake=$false; $script:kitImg=$false; $script:kitAvail=@()
function Detect-Kits {
  $script:kitLlm = [bool](Get-KitModels 'llm').Count
  $script:kitStt = [bool](Get-KitModels 'stt').Count
  $script:kitPya = [bool](Get-KitModels 'pya').Count
  $script:kitTts = [bool](Get-KitModels 'tts').Count
  $script:kitWake = [bool](Get-KitModels 'wake').Count
  $script:kitImg = [bool](Get-KitModels 'img').Count
  $script:kitAvail = @(); foreach ($k in $script:KitCatalog) { if ((Get-KitModels $k.id).Count) { $script:kitAvail += $k.id } }
}
function Show-Pkg-Panel([bool]$keepStatus=$false) {
  Detect-Kits
  if (-not $script:kitAvail.Count) { $ui.PkgPanel.Visibility='Collapsed'; return }
  $ui.PkgPanel.Visibility='Visible'; if (-not $keepStatus) { $ui.PkgStatus.Text='' }
  $ui.BtnPkg.IsEnabled=$true
  $ui.PkgLlm.IsEnabled=$script:kitLlm; $ui.PkgLlm.IsChecked=$script:kitLlm
  $ui.PkgStt.IsEnabled=$script:kitStt; $ui.PkgStt.IsChecked=$script:kitStt
  $ui.PkgPya.IsEnabled=$script:kitPya; $ui.PkgPya.IsChecked=$script:kitPya
  $ui.PkgTts.IsEnabled=$script:kitTts; $ui.PkgTts.IsChecked=$script:kitTts
  $ui.PkgWake.IsEnabled=$script:kitWake; $ui.PkgWake.IsChecked=$script:kitWake
  $ui.PkgImg.IsEnabled=$script:kitImg; $ui.PkgImg.IsChecked=$script:kitImg
}
$ui.BtnPkg.Add_Click({
  if ($sync.running) { return }
  $sel=@(); if ($ui.PkgLlm.IsChecked) { $sel+='llm' }; if ($ui.PkgStt.IsChecked) { $sel+='stt' }; if ($ui.PkgPya.IsChecked) { $sel+='pya' }; if ($ui.PkgTts.IsChecked) { $sel+='tts' }; if ($ui.PkgWake.IsChecked) { $sel+='wake' }; if ($ui.PkgImg.IsChecked) { $sel+='img' }
  if (-not $sel.Count) { $ui.PkgStatus.Text=T 'setup.p3.pkg_pick'; return }
  $ui.BtnPkg.IsEnabled=$false; $ui.PkgLlm.IsEnabled=$false; $ui.PkgStt.IsEnabled=$false; $ui.PkgPya.IsEnabled=$false; $ui.PkgTts.IsEnabled=$false; $ui.PkgWake.IsEnabled=$false; $ui.PkgImg.IsEnabled=$false
  $ui.PkgStatus.Foreground='#8B939F'; $ui.PkgStatus.Text=T 'setup.p3.pkg_starting'
  $cfg = if ($script:precfg) { $script:precfg } else { [ordered]@{} }
  $cfg.KITS = $sel
  # Modellwahl: vom Setup-Durchlauf uebernehmen, sonst Erst-Modell je Kit
  $km=@{}; foreach ($k in $sel) { $km[$k] = if ($script:kitModels -and $script:kitModels[$k]) { $script:kitModels[$k] } else { (Get-KitModels $k)[0].id } }
  $cfg.KITMODELS = $km
  Start-Core $cfg 'kits'
})
$ui.BtnCopy.Add_Click({ try { [Windows.Clipboard]::SetText($ui.DoneCmd.Text); $ui.BtnCopy.Content=T 'setup.p3.copied' } catch {} })
$ui.BtnAgain.Add_Click({ Set-Mode 'setup'; $ui.Feed.Children.Clear(); $ui.BtnStart.IsEnabled=$true; $ui.BtnStart.Content=T 'setup.p2.start'; $ui.FHost.Text=New-AutoHost; Show-Page 1 })
$ui.BtnClose.Add_Click({ $W.Close() })
if ($env:BARRA_TESTLOG) { $timer.Add_Tick({ try { [IO.File]::WriteAllText($env:BARRA_TESTLOG, $ui.Log.Text, [Text.Encoding]::UTF8) } catch {} }) }
# Dev-Hilfe: BARRA_TESTPAGE=pkg | model:<kitid> springt direkt auf die neuen Bildschirme (Screenshot-Abnahme)
if ($env:BARRA_TESTPAGE) {
  Detect-Kits
  if (-not $script:precfg) { $script:precfg = [ordered]@{ USER='test'; HOST='barra-test'; SSID='' } }
  if ($env:BARRA_TESTPAGE -eq 'pkg') { Init-Pkg-Screen; Show-Page 'pkg' }
  elseif ($env:BARRA_TESTPAGE -like 'model*') {
    $kid = if ($env:BARRA_TESTPAGE -match ':(\w+)') { $matches[1] } else { $script:kitAvail[0] }
    if ($kid) { $script:selKits=@($kid); $script:kitModels=@{}; $script:modelIdx=0; Show-Model-Screen }
  }
}
$W.Add_Closing({ $sync.cancel=$true; try { if ($script:ps) { $script:ps.BeginStop($null,$null) | Out-Null } } catch {} })   # BeginStop: Stop() blockiert den UI-Thread, wenn der Kern in einem adb-Wait steckt (Freeze 23.8.)
$W.ShowDialog() | Out-Null
