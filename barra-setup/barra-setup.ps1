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
        Title="barra Setup" Width="1060" Height="620" MinWidth="900" MinHeight="560" WindowStartupLocation="CenterScreen"
        Background="#0D1015" FontFamily="Segoe UI" FontSize="14">
  <Window.Resources>
    <Style TargetType="Button">
      <Setter Property="Background" Value="#3D6FE0"/><Setter Property="Foreground" Value="White"/><Setter Property="BorderThickness" Value="0"/>
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
  </Window.Resources>
  <Grid Margin="20">
    <Grid.RowDefinitions><RowDefinition Height="Auto"/><RowDefinition Height="*"/></Grid.RowDefinitions>
    <DockPanel Grid.Row="0" Margin="0,0,0,14">
      <TextBlock Text="barra" FontSize="30" FontWeight="SemiBold" Foreground="#F0F2F6"/>
      <TextBlock x:Name="Sub" Text="" FontSize="16" Foreground="#8B939F" VerticalAlignment="Bottom" Margin="0,0,0,5"/>
      <ComboBox x:Name="FLang" DockPanel.Dock="Right" Width="110" VerticalAlignment="Bottom" Margin="12,0,0,3"/>
      <TextBlock x:Name="Pager" Text="" Foreground="#8B939F" HorizontalAlignment="Right" VerticalAlignment="Bottom" Margin="0,0,0,5"/>
    </DockPanel>

    <!-- SEITE 1: Einrichten -->
    <Border Grid.Row="1" x:Name="Page1" Background="#161A21" CornerRadius="14" Padding="26">
      <Grid>
        <Grid.RowDefinitions><RowDefinition Height="Auto"/><RowDefinition Height="*"/><RowDefinition Height="Auto"/></Grid.RowDefinitions>
        <StackPanel Grid.Row="0"><TextBlock x:Name="P1Title" Text="" FontSize="22" FontWeight="SemiBold" Foreground="#F0F2F6"/><TextBlock x:Name="P1Intro" Text="" Margin="0,4,0,0" TextWrapping="Wrap"/></StackPanel>
        <Grid Grid.Row="1" Margin="0,8,0,0">
          <Grid.ColumnDefinitions><ColumnDefinition Width="*"/><ColumnDefinition Width="22"/><ColumnDefinition Width="*"/><ColumnDefinition Width="22"/><ColumnDefinition Width="*"/></Grid.ColumnDefinitions>
          <StackPanel Grid.Column="0">
            <TextBlock x:Name="LUser" Style="{StaticResource Lbl}"/><TextBox x:Name="FUser"/>
            <TextBlock x:Name="LPass" Style="{StaticResource Lbl}"/><PasswordBox x:Name="FPass"/>
            <TextBlock x:Name="LPass2" Style="{StaticResource Lbl}"/><PasswordBox x:Name="FPass2"/>
            <TextBlock x:Name="LHost" Style="{StaticResource Lbl}"/><TextBox x:Name="FHost"/>
            <TextBlock x:Name="LTz" Style="{StaticResource Lbl}"/><TextBox x:Name="FTz"/>
          </StackPanel>
          <StackPanel Grid.Column="2">
            <TextBlock x:Name="LSsid" Style="{StaticResource Lbl}"/>
            <DockPanel><Button x:Name="BtnWifi" Content="" DockPanel.Dock="Right" Padding="12,6" Margin="8,0,0,0" Background="#2A2F3A"/><TextBox x:Name="FSsid"/></DockPanel>
            <TextBlock x:Name="LPsk" Style="{StaticResource Lbl}"/><PasswordBox x:Name="FPsk"/>
            <TextBlock x:Name="LChg" Style="{StaticResource Lbl}"/>
            <DockPanel><TextBox x:Name="FChgS" Width="70" Text="50"/><TextBlock x:Name="LTo" Text="" VerticalAlignment="Center"/><TextBox x:Name="FChgE" Width="70" Text="55"/><TextBlock x:Name="LChgHint" Text="" FontSize="12" Foreground="#8B939F" VerticalAlignment="Center"/></DockPanel>
            <TextBlock x:Name="LDisp" Style="{StaticResource Lbl}"/><TextBox x:Name="FDisp" Width="90" HorizontalAlignment="Left" Text="20"/>
          </StackPanel>
          <StackPanel Grid.Column="4">
            <TextBlock x:Name="LKey" Style="{StaticResource Lbl}"/>
            <TextBox x:Name="FKey" Height="150" TextWrapping="Wrap" AcceptsReturn="True" VerticalScrollBarVisibility="Auto" FontSize="12"/>
            <TextBlock x:Name="FKeyHint" Text="" FontSize="12" Foreground="#8B939F" Margin="0,4,0,0"/>
          </StackPanel>
        </Grid>
        <DockPanel Grid.Row="2" Margin="0,16,0,0"><Button x:Name="BtnUnflash" Content="" DockPanel.Dock="Left" Background="#2A2F3A" Foreground="#C9CFD8" Padding="12,6"/><Button x:Name="BtnNext1" Content="" DockPanel.Dock="Right" HorizontalAlignment="Right"/><TextBlock x:Name="FErr" Foreground="#FF453A" VerticalAlignment="Center" Margin="14,0,0,0"/></DockPanel>
      </Grid>
    </Border>

    <!-- SEITE 2: Flashen -->
    <Grid Grid.Row="1" x:Name="Page2" Visibility="Collapsed">
      <Grid.ColumnDefinitions><ColumnDefinition Width="260"/><ColumnDefinition Width="*"/></Grid.ColumnDefinitions>
      <Grid.RowDefinitions><RowDefinition Height="*"/><RowDefinition Height="Auto"/></Grid.RowDefinitions>
      <Border Grid.Column="0" Background="#161A21" CornerRadius="14" Padding="16" Margin="0,0,14,0"><StackPanel x:Name="Steps"/></Border>
      <Border Grid.Column="1" Background="#161A21" CornerRadius="14" Padding="24">
        <Grid><Grid.RowDefinitions><RowDefinition Height="Auto"/><RowDefinition Height="Auto"/><RowDefinition Height="*"/><RowDefinition Height="Auto"/></Grid.RowDefinitions>
          <TextBlock x:Name="StepTitle" Grid.Row="0" Text="" FontSize="22" FontWeight="SemiBold" Foreground="#F0F2F6"/>
          <Border Grid.Row="1" x:Name="StateBox" Background="#1E232C" CornerRadius="10" Padding="14,10" Margin="0,12,0,0">
            <DockPanel><TextBlock x:Name="StateDot" Text="●" FontSize="18" Foreground="#3D6FE0" VerticalAlignment="Center" Margin="0,0,10,0"/>
              <StackPanel><TextBlock x:Name="StateText" Text="" FontSize="15" Foreground="#F0F2F6"/><ProgressBar x:Name="Prog" Height="6" Margin="0,8,0,0" Minimum="0" Maximum="100" Value="0" Foreground="#3D6FE0" Background="#0A0C10" BorderThickness="0" Visibility="Collapsed"/><TextBlock x:Name="ProgText" Text="" FontSize="12" Foreground="#8B939F" Margin="0,4,0,0"/></StackPanel>
            </DockPanel>
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
    <Border Grid.Row="1" x:Name="Page3" Background="#161A21" CornerRadius="14" Padding="30" Visibility="Collapsed">
      <StackPanel VerticalAlignment="Center" HorizontalAlignment="Center" MaxWidth="620">
        <TextBlock Text="✔" FontSize="56" Foreground="#34C759" HorizontalAlignment="Center"/>
        <TextBlock x:Name="DoneTitle" Text="" FontSize="26" FontWeight="SemiBold" Foreground="#F0F2F6" HorizontalAlignment="Center" Margin="0,8,0,0"/>
        <TextBlock x:Name="DoneSub" Text="" HorizontalAlignment="Center" Margin="0,6,0,18" TextWrapping="Wrap" TextAlignment="Center"/>
        <Border Background="#0A0C10" CornerRadius="10" Padding="16"><DockPanel><Button x:Name="BtnCopy" Content="" DockPanel.Dock="Right" Padding="12,6" Background="#2A2F3A" Margin="12,0,0,0"/><TextBox x:Name="DoneCmd" FontFamily="Consolas" FontSize="15" IsReadOnly="True" BorderThickness="0" Background="Transparent" Foreground="#F0F2F6" VerticalAlignment="Center"/></DockPanel></Border>
        <TextBlock x:Name="DoneHint" Text="" Margin="0,14,0,0" TextWrapping="Wrap" TextAlignment="Center" FontSize="13" Foreground="#8B939F"/>
        <StackPanel Orientation="Horizontal" HorizontalAlignment="Center" Margin="0,22,0,0"><Button x:Name="BtnAgain" Content="" Margin="0,0,12,0"/><Button x:Name="BtnClose" Content="" Background="#2A2F3A"/></StackPanel>
      </StackPanel>
    </Border>
  </Grid>
</Window>
"@
$W = [Windows.Markup.XamlReader]::Load((New-Object System.Xml.XmlNodeReader $xaml))
$ui=@{}; foreach ($n in 'Sub','Pager','FLang','Page1','Page2','Page3','P1Title','P1Intro','LUser','LPass','LPass2','LHost','LTz','LSsid','LPsk','LChg','LTo','LChgHint','LDisp','LKey','FUser','FPass','FPass2','FHost','FTz','FSsid','BtnWifi','FPsk','FKey','FKeyHint','FChgS','FChgE','FDisp','FErr','BtnNext1','BtnUnflash','Steps','StepTitle','StateBox','StateDot','StateText','Prog','ProgText','FeedScroll','Feed','ActionBox','ActionText','BtnYes','BtnNo','BtnStart','BtnLog','BtnBack','Status','LogBox','Log','DoneTitle','DoneSub','DoneCmd','BtnCopy','DoneHint','BtnAgain','BtnClose') { $ui[$n]=$W.FindName($n) }

# ---------- Sprache anwenden (alle statischen Texte; wird auch beim Dropdown-Wechsel gerufen) ----------
$script:mode='setup'
function Apply-UiLang {
  $script:stateWords = @{ ready=(T 'setup.state.ready'); run=(T 'setup.state.run'); wait_user=(T 'setup.state.wait_user'); wait_dev=(T 'setup.state.wait_dev'); ok=(T 'setup.state.ok'); fail=(T 'setup.state.fail') }
  $ui.P1Title.Text=T 'setup.p1.title'; $ui.P1Intro.Text=T 'setup.p1.intro'
  $ui.LUser.Text=T 'setup.p1.user'; $ui.LPass.Text=T 'setup.p1.pass'; $ui.LPass2.Text=T 'setup.p1.pass2'
  $ui.LHost.Text=T 'setup.p1.host'; $ui.LTz.Text=T 'setup.p1.tz'; $ui.LSsid.Text=T 'setup.p1.ssid'
  $ui.BtnWifi.Content=T 'setup.p1.wifi_btn'; $ui.LPsk.Text=T 'setup.p1.psk'; $ui.LChg.Text=T 'setup.p1.charge'
  $ui.LTo.Text=(T 'setup.p1.to')+'  '; $ui.LChgHint.Text=T 'setup.p1.charge_hint'; $ui.LDisp.Text=T 'setup.p1.disp'; $ui.LKey.Text=T 'setup.p1.key'
  $ui.BtnUnflash.Content=T 'setup.p1.unflash'; $ui.BtnUnflash.ToolTip=T 'setup.p1.unflash_tip'; $ui.BtnNext1.Content=T 'setup.p1.next'
  $ui.StepTitle.Text=T 'setup.p2.title0'; $ui.StateText.Text=$script:stateWords.ready
  $ui.BtnYes.Content=T 'setup.p2.done_btn'; $ui.BtnNo.Content=T 'setup.p2.cancel'
  $ui.BtnStart.Content=T 'setup.p2.start'; $ui.BtnLog.Content=T 'setup.p2.details_closed'; $ui.BtnBack.Content=T 'setup.p2.back'
  $ui.DoneTitle.Text=T 'setup.p3.done'; $ui.BtnCopy.Content=T 'setup.p3.copy'; $ui.BtnAgain.Content=T 'setup.p3.again'; $ui.BtnClose.Content=T 'setup.p3.close'
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
function Show-Page($n){ $ui.Page1.Visibility=if($n -eq 1){'Visible'}else{'Collapsed'}; $ui.Page2.Visibility=if($n -eq 2){'Visible'}else{'Collapsed'}; $ui.Page3.Visibility=if($n -eq 3){'Visible'}else{'Collapsed'}; $ui.Pager.Text=(T 'setup.pager' $n) }

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

$stateColors=@{ run='#3D6FE0'; wait_user='#FFD166'; wait_dev='#FF9F0A'; ok='#34C759'; fail='#FF453A' }
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
function Start-Core($precfg){
  $sync.done=$false; $sync.running=$true; $sync.cancel=$false
  $script:rs=[runspacefactory]::CreateRunspace(); $script:rs.ApartmentState='MTA'; $script:rs.Open()
  # ACHTUNG: PowerShell-Variablen sind case-insensitiv und der Kern setzt beim Laden $script:PreCfg/$script:Kit
  # in DERSELBEN Scope -> die injizierten Variablen brauchen eindeutige Namen (sonst genullt).
  $script:rs.SessionStateProxy.SetVariable('sync',$sync); $script:rs.SessionStateProxy.SetVariable('barraKitDir',$Kit); $script:rs.SessionStateProxy.SetVariable('barraPreCfgIn',$precfg); $script:rs.SessionStateProxy.SetVariable('barraModeIn',$script:mode); $script:rs.SessionStateProxy.SetVariable('barraLangIn',$script:lang)
  $script:ps=[powershell]::Create(); $script:ps.Runspace=$script:rs
  $script:ps.AddScript({
    . (Join-Path $barraKitDir 'barra-i18n.ps1')
    Initialize-BarraI18n -Dir (Join-Path $barraKitDir 'i18n') -Lang $barraLangIn
    . (Join-Path $barraKitDir 'barra-core.ps1')
    $script:Emit = { param($k,$d) $sync.q.Enqueue(@{ k=$k; d=$d }) }
    $script:PreCfg = $barraPreCfgIn
    # Answer/Cancel aus der GUI durchreichen
    $script:Answer = $sync.answerQ
    $t = [System.Threading.Thread]::CurrentThread
    try { if ($barraModeIn -eq 'unflash') { Run-Unflash } else { Run-Flash } } catch { $sync.q.Enqueue(@{ k='fail'; d=(T 'core.corefail' "$_") }) }
    $sync.done=$true
  }) | Out-Null
  $sync.answerQ=[System.Collections.Concurrent.ConcurrentQueue[string]]::new()
  $script:ps.BeginInvoke() | Out-Null
}
$script:askKind=''
function Send-Answer($a){ Write-LogFile "## Antwort: $a"; $sync.answerQ.Enqueue($a); $ui.ActionBox.Visibility='Collapsed' }

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
  $script:precfg = [ordered]@{ USER=$u; PASS=$p1; HOST=$h; TZ=$tz; SSID=$ssid; PSK=$psk; SSHKEY=($key -replace "`r?`n",' '); CHARGE_START=$ui.FChgS.Text; CHARGE_STOP=$ui.FChgE.Text; DISPLAY_TIMEOUT=$ui.FDisp.Text; LANG_UI=$script:lang }
  Save-Cfg @{ user=$u; host=$h; tz=$tz; ssid=$ssid; psk=$psk; key=$key; chgS=$ui.FChgS.Text; chgE=$ui.FChgE.Text; disp=$ui.FDisp.Text; lang=$script:lang }
  Show-Page 2
  Add-Feed (T 'setup.p2.saved' $u $h $(if($ssid){ T 'setup.p2.saved_wifi' $ssid } else { '' })) '#8B939F'
  Add-Feed (T 'setup.p2.connect_hint') '#C9CFD8'
})

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
      'done'  { if ($d.kind -eq 'unflash') { $ui.DoneTitle.Text=T 'setup.p3.u_title'; $ui.DoneSub.Text=T 'setup.p3.u_sub'; $ui.DoneCmd.Text=T 'setup.p3.u_cmd'; $ui.DoneHint.Text=T 'setup.p3.u_hint'; Show-Page 3; break }
                $ui.DoneTitle.Text=(T 'setup.p3.done_host' $d.host); $ui.DoneSub.Text=$(if($d.ip){ T 'setup.p3.reach' $d.ip }else{ T 'setup.p3.noip' }); $ui.DoneCmd.Text=$(if($d.ip){"ssh $($d.user)@$($d.ip)"}else{"ssh $($d.user)@<ip>"}); $ui.DoneHint.Text=T 'setup.p3.hint'; Show-Page 3 }
    }
    } catch { Write-LogFile "GUI-Fehler bei '$k': $_" }
  }
  if ($sync.done -and $sync.running) { $sync.running=$false; $ui.BtnStart.IsEnabled=$true; $ui.BtnStart.Content=T 'setup.p2.restart'; $ui.BtnBack.IsEnabled=$true; if ($script:lastState -ne 'ok') { $ui.Status.Text=T 'setup.p2.ended' } }
})
$timer.Start()
$ui.BtnCopy.Add_Click({ try { [Windows.Clipboard]::SetText($ui.DoneCmd.Text); $ui.BtnCopy.Content=T 'setup.p3.copied' } catch {} })
$ui.BtnAgain.Add_Click({ Set-Mode 'setup'; $ui.Feed.Children.Clear(); $ui.BtnStart.IsEnabled=$true; $ui.BtnStart.Content=T 'setup.p2.start'; $ui.FHost.Text=New-AutoHost; Show-Page 1 })
$ui.BtnClose.Add_Click({ $W.Close() })
if ($env:BARRA_TESTLOG) { $timer.Add_Tick({ try { [IO.File]::WriteAllText($env:BARRA_TESTLOG, $ui.Log.Text, [Text.Encoding]::UTF8) } catch {} }) }
$W.Add_Closing({ $sync.cancel=$true; try { if ($script:ps) { $script:ps.Stop() } } catch {} })
$W.ShowDialog() | Out-Null
